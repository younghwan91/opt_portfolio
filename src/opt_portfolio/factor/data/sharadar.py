"""
Sharadar 어댑터 — 직판 API (api.sharadar.com) 1차, Nasdaq Data Link 폴백

세 가지 수급 경로:
- **direct**: sharadar.com 직판 REST.
  `https://api.sharadar.com/v1.0/data/{table}?api_key=..&format=json&limit=..`
  커서가 없고 결과가 한도를 넘으면 일부만 돌려준다. **어느 쪽 끝을 주는지가
  ticker 필터 유무로 갈린다**: 필터가 있으면 `sort=date.asc` 가 선택까지
  지배해 *가장 오래된* N행을, 필터가 없으면 정렬과 무관하게 *가장 최근* N행을
  돌려준다 (둘 다 실측 확인).
  → 항상 티커 청크로 요청하고(필터 있음 상태를 강제), `from` 을 올리며
  과거→최신으로 마칭한다. 경계 날짜 행 중복은 스토어 업서트가 멱등 처리한다.
- **ndl**: Nasdaq Data Link datatables (커서 페이지네이션). 직판 장애 시 폴백.
- **CSV**: 벌크 다운로드 파일. 초기 전체 적재용 (수 GB).

직판 스키마는 2026-08-05 실계정으로 검증됨: 슬러그 전부 200, 모든 테이블의
기준 날짜 컬럼이 'date' 로 통일 (리네임으로 복원), 숫자는 문자열, DAILY 의
marketcap/ev 는 백만 달러 단위 (달러로 환산).

환경변수: SHARADAR_API_KEY (직판) / NASDAQ_DATA_LINK_API_KEY (폴백)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from opt_portfolio.factor.data.provider import normalize_columns, validate_pit_frame
from opt_portfolio.factor.data.schema import DEFAULT_DIMENSION, FILING_LAG_13F_DAYS

logger = logging.getLogger(__name__)

_NDL_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json"
_DIRECT_URL = "https://api.sharadar.com/v1.0/data/{table}"

#: NDL 테이블 코드 → 직판 슬러그. 2026-08-05 실계정으로 전수 검증됨 (전부 200).
_DIRECT_TABLES = {
    "SF1": "fundamentals",
    "SEP": "sep",
    "DAILY": "daily",
    "SF2": "sf2",
    "SF3A": "sf3a",
    "TICKERS": "tickers",
    "ACTIONS": "actions",
    "SP500": "sp500",
}

#: 이벤트 테이블의 히스토리 시작점. 무필터 요청은 `from` 이 없으면 **최신 N행**을
#: 돌려주므로(함정 4번), 전량 수집에는 항상 이 값을 넘긴다. 2026-08-11 실측:
#: from 을 주면 무필터에서도 sort=date.asc 가 정상 동작한다.
_EVENT_EPOCH = "1990-01-01"

#: 직판 윈도잉 페이지네이션의 정렬/커서 기준 날짜 컬럼.
#: 직판 API 는 모든 테이블의 기준 날짜 컬럼명을 'date' 로 통일했다
#: (SF1 의 datekey, SF2 의 filingdate, SF3A 의 calendardate 가 전부 date).
_DIRECT_PAGE_COL: dict[str, str | None] = {
    "SF1": "date",
    "SEP": "date",
    "DAILY": "date",
    "SF2": "date",
    "SF3A": "date",
    "TICKERS": None,  # 소형 테이블 — 단일 요청
    "ACTIONS": "date",
    "SP500": "date",
}

#: 직판 응답의 'date' → NDL/스토어 표준 컬럼명 복원 (실응답으로 검증)
_DIRECT_RENAME: dict[str, dict[str, str]] = {
    "SF1": {"date": "datekey"},
    "SF2": {"date": "filingdate"},
    "SF3A": {"date": "calendardate"},
}

#: 요청당 티커 개수 상한 — **벤더 하드 리밋** (2026-08-11 실측).
#: 초과 시 400 {"error":"Too many tickers","description":"ticker accepts at most
#: 30 tickers per request"}. 청크 크기는 이 값을 절대 넘을 수 없다.
MAX_TICKERS_PER_REQUEST = 30

#: 테이블별 티커 청크 크기 — 요청당 행수를 페이지 한도 아래로 유지하되,
#: 위 벤더 리밋을 넘지 않는다. (SEP 는 티커당 5년 ≈ 1,260행이므로 5개씩)
_DIRECT_CHUNK = {"SEP": 5, "DAILY": 5, "SF1": 30, "SF3A": 30, "SF2": 30, "TICKERS": 30}

#: 13F 집계(SF3A) 벤더 컬럼 → 표준 필드
_SF3A_RENAME = {"shrunits": "inst_shares", "shrholders": "inst_holders"}


class TransientAPIError(RuntimeError):
    """재시도 대상 (429 / 5xx)."""


class TruncatedDataError(RuntimeError):
    """
    페이지네이션이 전 구간을 받지 못하고 끝났다 — 조용히 넘어가면 안 되는 상황.

    이 저장소의 실데이터 버그는 전부 절단이 '성공' 로그와 함께 발생해서
    한참 뒤에야 드러났다. 부분 데이터로 백테스트를 돌리면 수익률이 조용히
    틀리므로, 잘린 것이 확실한 경로에서는 진행하지 않고 즉시 실패한다.
    """


def _ticker_param(tickers: list[str] | None) -> dict:
    """NDL 경로 전용 — 직판은 청크 단위로 ticker 를 직접 세팅한다."""
    return {"ticker": ",".join(tickers)} if tickers else {}


class SharadarProvider:
    """
    Args:
        api_key: 미지정 시 환경변수에서 읽는다.
        get_json: HTTP GET 주입 지점 — 테스트에서 가짜 응답으로 대체.
        page_size: datatables API 페이지 크기 (최대 10,000).
    """

    name = "sharadar"

    def __init__(
        self,
        api_key: str | None = None,
        get_json: Callable[[str, dict], dict] | None = None,
        page_size: int = 10_000,
        api: str = "direct",
        chunk_size: int | None = None,
    ) -> None:
        """
        Args:
            api: "direct" (sharadar.com, 기본) 또는 "ndl" (Nasdaq Data Link 폴백)
            chunk_size: 티커 청크 크기를 전 테이블 공통으로 덮어쓴다.
                기본값(`_DIRECT_CHUNK`)은 **5년 히스토리 기준**이라, 풀
                히스토리를 받으면 티커당 행수가 늘어 청크를 줄여야 한다.
        """
        if api not in ("direct", "ndl"):
            raise ValueError(f"api 는 'direct' 또는 'ndl' 이어야 합니다: {api!r}")
        self.api = api
        self.api_key = (
            api_key
            or os.environ.get("SHARADAR_API_KEY")
            or os.environ.get("NASDAQ_DATA_LINK_API_KEY")
            or ""
        )
        self._get_json = get_json or _default_get_json
        self.page_size = page_size
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------ API
    def fundamentals(
        self, since: str | None = None, tickers: list[str] | None = None
    ) -> Iterator[pd.DataFrame]:
        params: dict = {"dimension": DEFAULT_DIMENSION}
        if since:
            params["lastupdated.gte"] = since
        for chunk in self._paginate("SF1", params, tickers):
            frame = normalize_columns(chunk, "sharadar")
            validate_pit_frame(frame)
            yield frame

    def prices(
        self, since: str | None = None, tickers: list[str] | None = None
    ) -> Iterator[pd.DataFrame]:
        params: dict = {}
        if since:
            params["lastupdated.gte"] = since
        for chunk in self._paginate("SEP", params, tickers):
            yield normalize_columns(_drop_raw_close(chunk), "sharadar")

    def daily_metrics(
        self, since: str | None = None, tickers: list[str] | None = None
    ) -> Iterator[pd.DataFrame]:
        """
        DAILY 테이블 — 일별 marketcap/ev.

        ⚠️ DAILY 의 marketcap/ev 는 **백만 달러 단위**다 (SF1 은 달러 단위 —
        실응답으로 확인: AAPL 4,428,166.1 vs 4,508,288,143,800). 달러로
        환산하지 않으면 PER 등 배수가 10⁶배 틀어지므로 여기서 통일한다.
        """
        params: dict = {}
        if since:
            params["lastupdated.gte"] = since
        for chunk in self._paginate("DAILY", params, tickers):
            frame = normalize_columns(chunk, "sharadar")
            for col in ("mcap", "ev"):
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce") * 1e6
            yield frame

    def institutions(
        self, since: str | None = None, tickers: list[str] | None = None
    ) -> Iterator[pd.DataFrame]:
        """
        SF3A (13F 티커별 집계).

        SF3 에는 공시일 컬럼이 없다 — 분기말 + 45일 (법정 기한) 로 보정한다.
        실제 공시일보다 보수적(늦은) 가정이므로 look-ahead 방향으로는 안전하다.
        """
        params: dict = {}
        if since:
            params["calendardate.gte"] = since
        for chunk in self._paginate("SF3A", params, tickers):
            frame = chunk.rename(columns=_SF3A_RENAME)
            frame["datekey"] = pd.to_datetime(frame["calendardate"]) + pd.Timedelta(
                days=FILING_LAG_13F_DAYS
            )
            validate_pit_frame(frame)
            yield frame

    def insiders(
        self, since: str | None = None, tickers: list[str] | None = None
    ) -> Iterator[pd.DataFrame]:
        """
        SF2 (Form 4) → 분기 집계.

        거래 단위 데이터를 (ticker, 분기) 순매수 주식수로 합산한다.
        datekey = 분기말 + 3일 — 개별 신고는 분기 중에 도착하지만
        '분기 합계'는 분기가 끝나기 전엔 확정값이 아니다 (신고가 더 올 수 있음).
        Form 4 마감이 거래 후 2영업일이므로 +3일이면 전량 수집이 보장된다.
        """
        params: dict = {}
        if since:
            params["filingdate.gte"] = since
        for chunk in self._paginate("SF2", params, tickers):
            yield _aggregate_insiders(chunk)

    def actions(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        """
        기업 액션 — split / dividend / listed / delisted / spinoff (2026-08-11 실측).

        폐지일(delisted)이 **시점별 유니버스**의 근거이고, split 의 `value` 가
        수정주가 소급 재조정의 독립 검증 수단이다. 티커 필터 없이 전량을 받되
        `from` 을 반드시 넘긴다 — 없으면 최신 N행만 온다.
        """
        yield from self._paginate("ACTIONS", {"from": since or _EVENT_EPOCH})

    def sp500(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        """
        S&P500 구성종목 이력 — current / historical / added / removed (실측).

        `historical` 은 과거 시점의 구성종목 스냅샷이다. 2012-12-31 조회에
        YHOO 가 들어 있는 것을 확인했다 — 즉 편출 종목이 보존돼 있고,
        '당시 지수 편입 종목' 유니버스를 재구성할 수 있다.
        """
        yield from self._paginate("SP500", {"from": since or _EVENT_EPOCH})

    def tickers(self, tickers: list[str] | None = None) -> pd.DataFrame:
        # NDL 은 table=SF1 필터가 필요하지만, 직판은 이 필터에 빈 결과를
        # 반환한다 (table 컬럼 값 체계가 다름 — 실적재에서 확인)
        params: dict = {}
        if self.api == "ndl":
            params["table"] = "SF1"
        frames = list(self._paginate("TICKERS", params, tickers))
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        out = normalize_columns(raw, "sharadar")
        if "isdelisted" in out.columns and "is_delisted" not in out.columns:
            out = out.rename(columns={"isdelisted": "is_delisted"})
        return out

    def accessible_tickers(self) -> list[str]:
        """
        이 API 키로 실제 재무 데이터가 조회되는 티커 목록.

        구독 티어마다 유니버스가 다르므로(무료 = S&P500 현재 구성종목) 하드코딩
        대신 최근 분기 재무를 조회해 알아낸다.

        ⚠️ **생존편향 있음 — 유니버스 확정용으로 쓰면 안 된다.**
        최근 4개 분기 SF1 을 훑는 방식이라 상장폐지 종목은 원리적으로 잡히지
        않는다. 유료 플랜은 폐지 종목을 포함하므로(2026-08 벤더 확인), 이
        목록으로 적재하면 폐지 종목을 돈 주고 받아놓고 버리게 된다.
        전체 유니버스는 TICKERS 벌크 CSV 로 확정하라.
        """
        found: set[str] = set()
        for quarter_back in range(4):
            end = pd.Timestamp.today().normalize() - pd.DateOffset(months=3 * quarter_back)
            params = {"dimension": DEFAULT_DIMENSION, "fields": "ticker,calendardate"}
            params["from"] = str((end - pd.DateOffset(months=4)).date())
            for chunk in self._march_forward(
                _DIRECT_URL.format(table=_DIRECT_TABLES["SF1"]), params, "date"
            ):
                found.update(chunk["ticker"].astype(str))
            if found:
                break
        return sorted(found)

    # ------------------------------------------------------------------ CSV
    def load_csv(self, path: str | Path, kind: str) -> Iterator[pd.DataFrame]:
        """
        벌크 CSV 적재 (kind: fundamentals | prices | institutions | insiders).

        압축(zip/gz) 그대로 지원 — pandas 가 확장자로 처리한다.
        """
        readers: dict[str, Callable[[pd.DataFrame], Iterator[pd.DataFrame]]] = {
            "fundamentals": lambda df: iter([_csv_fundamentals(df)]),
            "prices": lambda df: iter([normalize_columns(_drop_raw_close(df), "sharadar")]),
            "institutions": lambda df: iter([_csv_institutions(df)]),
            "insiders": lambda df: iter([_aggregate_insiders(df)]),
            # 이벤트 테이블은 벤더 컬럼명이 이미 표준형이라 그대로 통과시킨다
            "actions": lambda df: iter([df]),
            "sp500": lambda df: iter([df]),
            "tickers": lambda df: iter([_csv_tickers(df)]),
            "daily": lambda df: iter([_csv_daily(df)]),
        }
        if kind not in readers:
            raise ValueError(f"알 수 없는 CSV 종류 '{kind}'. 지원: {sorted(readers)}")
        for chunk in pd.read_csv(path, chunksize=200_000):
            yield from readers[kind](chunk)

    # ------------------------------------------------------------------ 내부
    def _paginate(
        self, table: str, params: dict, tickers: list[str] | None = None
    ) -> Iterator[pd.DataFrame]:
        if self.api == "direct":
            yield from self._paginate_direct(table, params, tickers)
        else:
            yield from self._paginate_ndl(table, {**params, **_ticker_param(tickers)})

    def _paginate_direct(
        self, table: str, params: dict, tickers: list[str] | None = None
    ) -> Iterator[pd.DataFrame]:
        """티커 청크 × 날짜 역방향 마칭 — 전 기간 수집을 보장한다."""
        slug = _DIRECT_TABLES.get(table, table.lower())
        url = _DIRECT_URL.format(table=slug)
        date_col = _DIRECT_PAGE_COL.get(table)
        rename = _DIRECT_RENAME.get(table, {})

        if tickers:
            # 벤더 하드 리밋을 넘는 값은 조용히 400 이 되므로 여기서 자른다
            size = min(self.chunk_size or _DIRECT_CHUNK.get(table, 30), MAX_TICKERS_PER_REQUEST)
            groups: list[list[str] | None] = [
                tickers[i : i + size] for i in range(0, len(tickers), size)
            ]
        else:
            groups = [None]

        for group in groups:
            query = dict(params)
            if group:
                query["ticker"] = ",".join(group)
            for frame in self._march_forward(url, query, date_col, table):
                yield frame.rename(columns=rename)

    def _march_forward(
        self, url: str, params: dict, date_col: str | None, table: str = "?"
    ) -> Iterator[pd.DataFrame]:
        """
        과거 → 최신 방향 페이지네이션.

        ticker 필터가 걸린 요청에서 `sort=date.asc` 는 선택까지 지배하므로
        첫 페이지가 가장 오래된 구간이다. 페이지가 한도만큼 차면 그 페이지의
        마지막 날짜부터 다음 페이지를 이어받는다 (경계 날짜가 페이지 중간에서
        잘릴 수 있어 +1일 하지 않고 같은 날짜부터 다시 받는다 — 중복은
        스토어가 멱등 처리).
        """
        from_date: str | None = params.pop("from", None)
        for _ in range(500):  # 무한루프 방지
            query = {
                **params,
                "api_key": self.api_key,
                "format": "json",
                "limit": self.page_size,
            }
            if date_col:
                query["sort"] = f"{date_col}.asc"
                if from_date:
                    query["from"] = from_date
            frame = _parse_direct_payload(self._fetch(url, query))
            if frame.empty:
                return
            yield frame
            if date_col is None:
                # 커서가 없어 이어받을 수단이 없다. 한도만큼 찼다면 뒷부분이
                # 있는지조차 확인할 수 없으므로 절단으로 간주한다 —
                # 유료 플랜의 TICKERS(폐지 포함 ~18,000종목)가 여기 걸린다.
                if len(frame) >= self.page_size:
                    raise TruncatedDataError(
                        f"{table} 는 날짜 커서가 없어 단일 요청으로 받는데 응답이 "
                        f"한도({self.page_size})를 채웠습니다 — 전량이 아닐 수 "
                        f"있습니다. page_size 를 올리거나 티커 청크로 나눠 요청하세요."
                    )
                return
            if len(frame) < self.page_size:
                return
            newest = str(pd.to_datetime(frame[date_col]).max().date())
            if newest == from_date:
                # 한 날짜의 행수가 페이지 한도를 넘어 전진이 불가능하다.
                # 여기서 멈추면 newest 이후 구간이 통째로 누락된다.
                raise TruncatedDataError(
                    f"페이지 전진 불가 — {newest} 하루 행수가 한도({self.page_size})를 "
                    f"넘어 이후 구간을 받을 수 없습니다. 티커 청크를 줄이세요 "
                    f"(대상: {params.get('ticker', '<no ticker>')})"
                )
            from_date = newest
        raise TruncatedDataError(
            f"페이지 상한(500) 도달 — {from_date} 이후 구간이 누락됐습니다. "
            f"티커 청크를 줄이거나 기간을 나눠 요청하세요 (대상: {params})"
        )

    def _paginate_ndl(self, table: str, params: dict) -> Iterator[pd.DataFrame]:
        """Nasdaq Data Link datatables — 커서 페이지네이션 (폴백 경로)."""
        cursor: str | None = None
        while True:
            query = {
                **params,
                "api_key": self.api_key,
                "qopts.per_page": self.page_size,
            }
            if cursor:
                query["qopts.cursor_id"] = cursor
            payload = self._fetch(_NDL_URL.format(table=table), query)
            if not isinstance(payload, dict):
                raise ValueError("NDL 응답은 dict 여야 합니다 (datatable 래퍼)")
            dt = payload.get("datatable", {})
            columns = [c["name"] for c in dt.get("columns", [])]
            rows = dt.get("data", [])
            if rows:
                yield pd.DataFrame(rows, columns=columns)
            cursor = payload.get("meta", {}).get("next_cursor_id")
            if not cursor:
                return

    @retry(
        retry=retry_if_exception_type(TransientAPIError),
        wait=wait_exponential(multiplier=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _fetch(self, url: str, params: dict) -> dict | list:
        payload: dict | list = self._get_json(url, params)
        return payload


def _parse_direct_payload(payload: dict | list) -> pd.DataFrame:
    """
    직판 JSON 응답 → DataFrame.

    문서에 응답 스키마가 명시돼 있지 않아 두 관례를 모두 받는다:
    레코드 배열([{...}, ...]) 또는 {columns: [...], data: [[...]]}.
    """
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if "data" in payload:
        columns = payload.get("columns")
        names = [c["name"] if isinstance(c, dict) else c for c in columns] if columns else None
        return pd.DataFrame(payload["data"], columns=names)
    raise ValueError(f"해석할 수 없는 직판 응답 형식: {type(payload).__name__}")


def _default_get_json(url: str, params: dict) -> dict | list:
    import requests

    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientAPIError(f"HTTP {resp.status_code}: {url}")
    resp.raise_for_status()
    payload: dict | list = resp.json()
    return payload


def _drop_raw_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    SEP 에는 close(원시)와 closeadj 가 공존한다. normalize 가 closeadj→close 로
    바꾸면 컬럼명이 중복되므로, 표준 스키마에 없는 원시 close 를 먼저 버린다.
    """
    if "closeadj" in df.columns and "close" in df.columns:
        return df.drop(columns=["close"])
    return df


def _csv_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[df.get("dimension", DEFAULT_DIMENSION) == DEFAULT_DIMENSION]
    frame = normalize_columns(frame, "sharadar")
    validate_pit_frame(frame)
    return frame


def _csv_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    DAILY 벌크 CSV → 정규화.

    API 경로(`daily_metrics`)와 **같은 백만 달러 → 달러 환산**을 적용한다.
    이게 빠지면 mcap/ev 가 10⁶배 작아져 PER·EV 배수가 전부 틀어지고,
    시총 유니버스 필터가 모든 종목을 소형주로 오인한다.
    """
    frame = normalize_columns(df, "sharadar")
    for col in ("mcap", "ev"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce") * 1e6
    return frame


def _csv_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    TICKERS 벌크 CSV → 정규화 메타.

    CSV 는 `isdelisted`, 스토어는 `is_delisted` 를 쓴다. API 경로만 리네임하고
    있어서 벌크로 적재하면 폐지 여부가 통째로 NULL 이 된다 — 폐지 종목을 사놓고
    어느 것이 폐지됐는지 모르게 되므로, 생존편향 제거가 여기서 무산된다.
    """
    out = normalize_columns(df, "sharadar")
    if "isdelisted" in out.columns and "is_delisted" not in out.columns:
        out = out.rename(columns={"isdelisted": "is_delisted"})
    return out


def _csv_institutions(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.rename(columns=_SF3A_RENAME)
    frame["datekey"] = pd.to_datetime(frame["calendardate"]) + pd.Timedelta(
        days=FILING_LAG_13F_DAYS
    )
    return frame


def _aggregate_insiders(chunk: pd.DataFrame) -> pd.DataFrame:
    """SF2 거래 단위 → (ticker, 분기) 순매수 주식수. datekey = 분기말 + 3일."""
    df = chunk.copy()
    df["filingdate"] = pd.to_datetime(df["filingdate"])
    df["calendardate"] = df["filingdate"] + pd.offsets.QuarterEnd(0)
    # transactionshares: 매수 양수 / 매도 음수 (Sharadar 규약).
    # 직판 JSON 은 숫자를 문자열로 주므로 합산 전 강제 변환한다.
    df["transactionshares"] = pd.to_numeric(df["transactionshares"], errors="coerce")
    grouped = (
        df.groupby(["ticker", "calendardate"])
        .agg(insider_net_shares=("transactionshares", "sum"))
        .reset_index()
    )
    grouped["datekey"] = grouped["calendardate"] + pd.Timedelta(days=3)
    validate_pit_frame(grouped)
    return grouped
