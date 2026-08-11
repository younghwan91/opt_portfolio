"""
PIT 스토어 — DuckDB 기반 bitemporal 저장소

테이블 구조 (전부 정규화 필드명 — 벤더 이름은 어댑터에서 끝난다):

| 테이블 | 키 | 내용 |
|---|---|---|
| fundamentals | (ticker, calendardate, dimension) | SF1 분기 재무 + datekey |
| institutions | (ticker, calendardate) | 13F 집계 + datekey (분기말+45일) |
| insiders | (ticker, calendardate) | 내부자 분기 집계 + datekey (분기말+3일) |
| estimates | (ticker, calendardate) | 애널리스트 추정치 + datekey |
| prices | (ticker, date) | 일별 가격·거래량·시총 (SEP+DAILY 컬럼 병합) |
| tickers | (ticker) | 섹터·소재지·유형 메타 |

13F 와 내부자를 다른 테이블에 두는 이유: 같은 분기라도 공시 지연이
다르다 (+45일 vs +3일). 한 테이블에 datekey 하나로 합치면 둘 중 하나의
가용 시점이 왜곡된다 — 실데이터 적재에서 확인된 설계 결함의 수정.

퀀트 관점:
- 같은 (ticker, calendardate) 에 공시가 여러 번 오면 (정정공시)
  **최초 datekey 의 값**을 유지한다 — 시장이 처음 본 숫자가 백테스트가
  봐야 하는 숫자다. 정정치를 쓰면 look-ahead 다.
- build_context() 는 소스별 datekey 를 분리해 넘긴다. 13F(+45일)와
  실적공시의 지연 차이가 표현식 트리의 avail 전파로 이어진다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from opt_portfolio.factor.data import schema
from opt_portfolio.factor.dsl.context import PanelContext

logger = logging.getLogger(__name__)


def _quarterly_fields(sources: tuple[str, ...]) -> list[str]:
    return sorted(
        s.name for s in schema.FIELDS.values() if s.grid == "quarterly" and s.source in sources
    )


FUND_FIELDS = _quarterly_fields(("SF1",))
INSTITUTION_FIELDS = _quarterly_fields(("SF3",))
INSIDER_FIELDS = _quarterly_fields(("SF2",))
ESTIMATE_FIELDS = _quarterly_fields(("FMP",))
PRICE_FIELDS = sorted(
    s.name for s in schema.FIELDS.values() if s.grid == "daily" and s.kind in ("price", "stock")
)
META_FIELDS = sorted(s.name for s in schema.FIELDS.values() if s.kind == "meta") + ["name"]


class PITStore:
    """벤더 중립 point-in-time 저장소."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = duckdb.connect(self.path)
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> PITStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ DDL
    def _init_schema(self) -> None:
        def cols(fields: list[str]) -> str:
            return ", ".join(f'"{f}" DOUBLE' for f in fields)

        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker VARCHAR NOT NULL,
                calendardate DATE NOT NULL,
                datekey DATE NOT NULL,
                dimension VARCHAR NOT NULL DEFAULT 'ARQ',
                {cols(FUND_FIELDS)}
            )
            """
        )
        for table, fields in (
            ("institutions", INSTITUTION_FIELDS),
            ("insiders", INSIDER_FIELDS),
        ):
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    ticker VARCHAR NOT NULL,
                    calendardate DATE NOT NULL,
                    datekey DATE NOT NULL,
                    {cols(fields)}
                )
                """
            )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS estimates (
                ticker VARCHAR NOT NULL,
                calendardate DATE NOT NULL,
                datekey DATE NOT NULL,
                {cols(ESTIMATE_FIELDS)}
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS prices (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                {cols(PRICE_FIELDS)}
            )
            """
        )
        meta_cols = ", ".join(f'"{f}" VARCHAR' for f in META_FIELDS)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS tickers (ticker VARCHAR NOT NULL, {meta_cols})"
        )

    # ---------------------------------------------------------------- 업서트
    def upsert_fundamentals(self, df: pd.DataFrame) -> int:
        """정규화된 분기 재무를 업서트. 필수 컬럼: ticker/calendardate/datekey."""
        return self._upsert(
            "fundamentals",
            df,
            keys=("ticker", "calendardate", "dimension"),
            fields=FUND_FIELDS,
            defaults={"dimension": schema.DEFAULT_DIMENSION},
        )

    def upsert_institutions(self, df: pd.DataFrame) -> int:
        return self._upsert(
            "institutions", df, keys=("ticker", "calendardate"), fields=INSTITUTION_FIELDS
        )

    def upsert_insiders(self, df: pd.DataFrame) -> int:
        return self._upsert("insiders", df, keys=("ticker", "calendardate"), fields=INSIDER_FIELDS)

    def upsert_estimates(self, df: pd.DataFrame) -> int:
        return self._upsert(
            "estimates", df, keys=("ticker", "calendardate"), fields=ESTIMATE_FIELDS
        )

    def upsert_prices(self, df: pd.DataFrame) -> int:
        return self._upsert(
            "prices",
            df,
            keys=("ticker", "date"),
            fields=PRICE_FIELDS,
            has_datekey=False,
            merge_fields=True,  # SEP(가격)와 DAILY(시총)가 같은 행의 다른 컬럼을 채운다
        )

    # -------------------------------------------------- 수정주가 소급 재조정
    def detect_adjustment_factors(
        self, incoming: pd.DataFrame, tol: float = 1e-4
    ) -> dict[str, float]:
        """
        겹치는 날짜의 수정주가 비율 → 종목별 재조정 계수.

        벤더의 `closeadj` 는 분할·배당이 생기면 과거 전체가 다시 계산된다.
        풀 히스토리를 받아두고 최근 구간만 증분 적재하면, 스토어의 옛 계수
        구간과 새로 받은 구간 사이 경계에 **가짜 수익률**이 생긴다.

        겹치는 날짜에서 (새 값 / 옛 값) 이 1 이 아니면 재조정이 일어난 것이고,
        그 비율이 곧 계수다. 비율이 날짜마다 다르면(분할이 아니라 데이터 정정)
        판단 근거가 없으므로 보고하지 않는다 — 추측으로 과거를 고치지 않는다.

        Returns:
            {ticker: factor} — 1 에서 유의하게 벗어난 종목만. 겹침이 없으면 제외.
        """
        if incoming.empty or "close" not in incoming.columns:
            return {}
        new = incoming[["ticker", "date", "close"]].dropna()
        if new.empty:
            return {}
        new = new.assign(date=pd.to_datetime(new["date"]))

        # 들어온 티커·날짜 범위로 좁힌다 — 전체 스캔은 풀 히스토리에서 감당 안 된다
        names = sorted(set(new["ticker"].astype(str)))
        placeholders = ",".join("?" * len(names))
        stored = self.conn.execute(
            f"SELECT ticker, date, close FROM prices "  # noqa: S608 (placeholders 만 보간)
            f"WHERE close IS NOT NULL AND ticker IN ({placeholders}) "
            f"AND date BETWEEN ? AND ?",
            [*names, new["date"].min().date(), new["date"].max().date()],
        ).fetch_df()
        if stored.empty:
            return {}
        stored["date"] = pd.to_datetime(stored["date"])

        merged = stored.merge(new, on=["ticker", "date"], suffixes=("_old", "_new"))
        merged = merged[merged["close_old"] > 0]
        if merged.empty:
            return {}
        merged["ratio"] = merged["close_new"] / merged["close_old"]

        factors: dict[str, float] = {}
        for ticker, group in merged.groupby("ticker"):
            ratios = group["ratio"]
            if ratios.std(ddof=0) > tol:  # 일관된 배수가 아니면 분할이 아니다
                continue
            factor = float(ratios.mean())
            if abs(factor - 1.0) > tol:
                factors[str(ticker)] = factor
        return factors

    def rescale_prices(self, factors: dict[str, float]) -> int:
        """
        저장된 수정주가에 계수를 곱해 재조정 이전 구간을 새 기준에 맞춘다.

        `close`(=closeadj) 만 대상이다 — `closeunadj`·`open`/`high`/`low` 는
        벤더가 주는 원시값이라 재조정되지 않는다.
        """
        if not factors:
            return 0
        total = 0
        for ticker, factor in factors.items():
            cur = self.conn.execute(
                "UPDATE prices SET close = close * ? WHERE ticker = ? AND close IS NOT NULL",
                [factor, ticker],
            )
            row = cur.fetchone() if cur.description else None
            total += int(row[0]) if row else 0
            logger.warning(
                "수정주가 소급 재조정: %s × %.6f — 벤더가 과거를 다시 계산했습니다",
                ticker,
                factor,
            )
        return total

    def upsert_tickers(self, df: pd.DataFrame) -> int:
        return self._upsert(
            "tickers",
            df,
            keys=("ticker",),
            fields=META_FIELDS,
            has_datekey=False,
            string_fields=True,
        )

    def _upsert(
        self,
        table: str,
        df: pd.DataFrame,
        *,
        keys: tuple[str, ...],
        fields: list[str],
        defaults: dict[str, str] | None = None,
        has_datekey: bool = True,
        string_fields: bool = False,
        merge_fields: bool = False,
    ) -> int:
        if df.empty:
            return 0
        frame = df.copy()
        for col, value in (defaults or {}).items():
            if col not in frame.columns:
                frame[col] = value

        required = list(keys) + (["datekey"] if has_datekey else [])
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{table} 업서트에 필수 컬럼 누락: {missing}")

        # 중복 컬럼 방어 — 어댑터 버그가 있어도 스토어는 깨지지 않게
        frame = frame.loc[:, ~frame.columns.duplicated()]
        present_fields = [f for f in fields if f in frame.columns]
        all_cols = required + present_fields
        frame = frame[all_cols]
        if not string_fields:
            for f in present_fields:
                frame[f] = pd.to_numeric(frame[f], errors="coerce")

        # 같은 키의 재공시(정정)는 최초 datekey 를 유지한다 — PIT 원칙
        sort_cols = required if has_datekey else list(keys)
        frame = (
            frame.sort_values(sort_cols)
            .drop_duplicates(subset=list(keys), keep="first")
            .reset_index(drop=True)
        )

        self.conn.register("_incoming", frame)
        key_match = " AND ".join(f"t.{k} = i.{k}" for k in keys)
        col_list = ", ".join(f'"{c}"' for c in all_cols)
        # 없는 행 삽입. PIT 테이블은 기존 행 유지(최초 공시 우선),
        # merge_fields 테이블은 기존 행의 빈 컬럼을 추가로 채운다.
        result = self.conn.execute(
            f"""
            INSERT INTO {table} ({col_list})
            SELECT {col_list} FROM _incoming i
            WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {key_match})
            """
        ).fetchone()
        if merge_fields and present_fields:
            set_clause = ", ".join(f'"{f}" = COALESCE(i."{f}", t."{f}")' for f in present_fields)
            self.conn.execute(
                f"UPDATE {table} t SET {set_clause} FROM _incoming i WHERE {key_match}"
            )
        self.conn.unregister("_incoming")
        count = int(result[0]) if result else 0  # DuckDB INSERT 는 삽입 행수를 반환
        logger.info("%s: %d행 삽입 (%d행 수신)", table, count, len(df))
        return count

    # ------------------------------------------------------------ 컨텍스트 조립
    def build_context(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        tickers: list[str] | None = None,
        benchmark: str | None = None,
    ) -> PanelContext:
        """
        저장소 → PanelContext.

        Args:
            benchmark: 지정 시 해당 티커의 일별 수익률을
                meta['benchmark_return'] 으로 넣는다 (베타 팩터·타이밍용).
        """
        quarterly: dict[str, pd.DataFrame] = {}
        avail_by_source: dict[str, pd.DataFrame] = {}

        fund, fund_avail = self._load_quarterly(
            "fundamentals", FUND_FIELDS, tickers, extra="dimension = 'ARQ'"
        )
        quarterly.update(fund)
        if fund_avail is not None:
            avail_by_source["SF1"] = fund_avail

        inst, inst_avail = self._load_quarterly("institutions", INSTITUTION_FIELDS, tickers)
        quarterly.update(inst)
        if inst_avail is not None:
            avail_by_source["SF3"] = inst_avail

        ins, ins_avail = self._load_quarterly("insiders", INSIDER_FIELDS, tickers)
        quarterly.update(ins)
        if ins_avail is not None:
            avail_by_source["SF2"] = ins_avail

        est, est_avail = self._load_quarterly("estimates", ESTIMATE_FIELDS, tickers)
        quarterly.update(est)
        if est_avail is not None:
            avail_by_source["FMP"] = est_avail

        daily = self._load_prices(start, end, tickers)
        meta = self._load_meta()

        calendar = None
        for frame in daily.values():
            calendar = pd.DatetimeIndex(frame.index)
            break

        if benchmark and "close" in daily and benchmark in daily["close"].columns:
            meta["benchmark_return"] = daily["close"][benchmark].pct_change(fill_method=None)

        return PanelContext(
            quarterly=quarterly,
            availability=fund_avail,
            availability_by_source=avail_by_source,
            daily=daily,
            meta=meta,
            calendar=calendar,
        )

    def _load_quarterly(
        self,
        table: str,
        fields: list[str],
        tickers: list[str] | None,
        extra: str | None = None,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
        where, params = self._where(tickers=tickers, extra=extra)
        raw = self.conn.execute(
            f"SELECT * FROM {table} {where} ORDER BY ticker, calendardate", params
        ).df()
        if raw.empty:
            return {}, None

        raw["calendardate"] = pd.to_datetime(raw["calendardate"])
        raw["datekey"] = pd.to_datetime(raw["datekey"])

        frames = {}
        for f in fields:
            if f in raw.columns and raw[f].notna().any():
                frames[f] = raw.pivot_table(
                    index="calendardate", columns="ticker", values=f, aggfunc="first"
                )
        avail = raw.pivot_table(
            index="calendardate", columns="ticker", values="datekey", aggfunc="min"
        )
        return frames, avail

    def _load_prices(
        self,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
        tickers: list[str] | None,
    ) -> dict[str, pd.DataFrame]:
        clauses, params = [], []
        if tickers:
            clauses.append(f"ticker IN ({', '.join('?' * len(tickers))})")
            params.extend(tickers)
        if start is not None:
            clauses.append("date >= ?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            clauses.append("date <= ?")
            params.append(str(pd.Timestamp(end).date()))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        raw = self.conn.execute(f"SELECT * FROM prices {where} ORDER BY date", params).df()
        if raw.empty:
            return {}
        raw["date"] = pd.to_datetime(raw["date"])

        return {
            f: raw.pivot_table(index="date", columns="ticker", values=f, aggfunc="first")
            for f in PRICE_FIELDS
            if f in raw.columns and raw[f].notna().any()
        }

    def _load_meta(self) -> dict[str, pd.Series]:
        raw = self.conn.execute("SELECT * FROM tickers").df()
        if raw.empty:
            return {}
        raw = raw.set_index("ticker")
        return {f: raw[f] for f in META_FIELDS if f in raw.columns}

    @staticmethod
    def _where(tickers: list[str] | None, extra: str | None) -> tuple[str, list[str]]:
        clauses, params = [], []
        if tickers:
            clauses.append(f"ticker IN ({', '.join('?' * len(tickers))})")
            params.extend(tickers)
        if extra:
            clauses.append(extra)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params

    # ------------------------------------------------------------------ 상태
    def known_tickers(self) -> list[str]:
        """가격 또는 재무 데이터가 실제로 적재된 티커 — 메타 수집 대상."""
        rows = self.conn.execute(
            "SELECT ticker FROM prices UNION SELECT ticker FROM fundamentals ORDER BY ticker"
        ).fetchall()
        return [r[0] for r in rows]

    def coverage(self) -> pd.DataFrame:
        """테이블별 행수·기간 — CLI status 와 수동 점검용."""
        rows = []
        for table, date_col in [
            ("fundamentals", "calendardate"),
            ("institutions", "calendardate"),
            ("insiders", "calendardate"),
            ("estimates", "calendardate"),
            ("prices", "date"),
        ]:
            r = self.conn.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT ticker), MIN({date_col}), "
                f"MAX({date_col}) FROM {table}"
            ).fetchone() or (0, 0, None, None)
            rows.append(
                {
                    "table": table,
                    "rows": r[0],
                    "tickers": r[1],
                    "first": r[2],
                    "last": r[3],
                }
            )
        n_meta = self.conn.execute("SELECT COUNT(*) FROM tickers").fetchone()
        rows.append(
            {
                "table": "tickers",
                "rows": n_meta[0] if n_meta else 0,
                "tickers": n_meta[0] if n_meta else 0,
                "first": None,
                "last": None,
            }
        )
        return pd.DataFrame(rows)
