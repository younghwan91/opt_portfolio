"""
Sharadar 어댑터 — 직판 API (api.sharadar.com) 1차, Nasdaq Data Link 폴백

세 가지 수급 경로:
- **direct**: sharadar.com 직판 REST.
  `https://api.sharadar.com/v1.0/data/{table}?api_key=..&format=json&limit=..`
  커서가 없으므로 날짜 오름차순 정렬 + `from=` 윈도잉으로 페이지네이션한다.
  경계 날짜 행이 페이지 간 중복될 수 있으나 스토어의 키 기반 업서트가 걸러낸다.
- **ndl**: Nasdaq Data Link datatables (커서 페이지네이션). 직판 장애 시 폴백.
- **CSV**: 벌크 다운로드 파일. 초기 전체 적재용 (수 GB).

⚠️ 직판 테이블 슬러그는 문서에서 `fundamentals` 만 확인됨 — 나머지는
   추정이며 무료 티어(다우 30)로 구독 전 검증한다. 틀리면 _DIRECT_TABLES
   상수 한 줄씩만 고치면 된다.

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

#: NDL 테이블 코드 → 직판 슬러그. 'fundamentals' 만 공식 문서에서 확인됨,
#: 나머지는 무료 티어로 검증 후 필요시 수정 (docs/factor-system/04 §4).
_DIRECT_TABLES = {
    "SF1": "fundamentals",
    "SEP": "sep",
    "DAILY": "daily",
    "SF2": "sf2",
    "SF3A": "sf3a",
    "TICKERS": "tickers",
}

#: 직판 윈도잉 페이지네이션의 정렬/커서 기준 날짜 컬럼
_DIRECT_PAGE_COL = {
    "SF1": "datekey",
    "SEP": "date",
    "DAILY": "date",
    "SF2": "filingdate",
    "SF3A": "calendardate",
    "TICKERS": None,  # 소형 테이블 — 단일 요청
}

#: 13F 집계(SF3A) 벤더 컬럼 → 표준 필드
_SF3A_RENAME = {"shrunits": "inst_shares", "shrholders": "inst_holders"}


class TransientAPIError(RuntimeError):
    """재시도 대상 (429 / 5xx)."""


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
    ) -> None:
        """
        Args:
            api: "direct" (sharadar.com, 기본) 또는 "ndl" (Nasdaq Data Link 폴백)
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

    # ------------------------------------------------------------------ API
    def fundamentals(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        params = {"dimension": DEFAULT_DIMENSION}
        if since:
            params["lastupdated.gte"] = since
        for chunk in self._paginate("SF1", params):
            frame = normalize_columns(chunk, "sharadar")
            validate_pit_frame(frame)
            yield frame

    def prices(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        params: dict = {}
        if since:
            params["lastupdated.gte"] = since
        for chunk in self._paginate("SEP", params):
            yield normalize_columns(chunk, "sharadar")

    def daily_metrics(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        """DAILY 테이블 — 일별 marketcap/ev. prices 와 같은 스토어 테이블로 합류."""
        params: dict = {}
        if since:
            params["lastupdated.gte"] = since
        for chunk in self._paginate("DAILY", params):
            yield normalize_columns(chunk, "sharadar")

    def institutions(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        """
        SF3A (13F 티커별 집계).

        SF3 에는 공시일 컬럼이 없다 — 분기말 + 45일 (법정 기한) 로 보정한다.
        실제 공시일보다 보수적(늦은) 가정이므로 look-ahead 방향으로는 안전하다.
        """
        params: dict = {}
        if since:
            params["calendardate.gte"] = since
        for chunk in self._paginate("SF3A", params):
            frame = chunk.rename(columns=_SF3A_RENAME)
            frame["datekey"] = pd.to_datetime(frame["calendardate"]) + pd.Timedelta(
                days=FILING_LAG_13F_DAYS
            )
            validate_pit_frame(frame)
            yield frame

    def insiders(self, since: str | None = None) -> Iterator[pd.DataFrame]:
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
        for chunk in self._paginate("SF2", params):
            yield _aggregate_insiders(chunk)

    def tickers(self) -> pd.DataFrame:
        frames = list(self._paginate("TICKERS", {"table": "SF1"}))
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        out = normalize_columns(raw, "sharadar")
        if "isdelisted" in out.columns and "is_delisted" not in out.columns:
            out = out.rename(columns={"isdelisted": "is_delisted"})
        return out

    # ------------------------------------------------------------------ CSV
    def load_csv(self, path: str | Path, kind: str) -> Iterator[pd.DataFrame]:
        """
        벌크 CSV 적재 (kind: fundamentals | prices | institutions | insiders).

        압축(zip/gz) 그대로 지원 — pandas 가 확장자로 처리한다.
        """
        readers: dict[str, Callable[[pd.DataFrame], Iterator[pd.DataFrame]]] = {
            "fundamentals": lambda df: iter([_csv_fundamentals(df)]),
            "prices": lambda df: iter([normalize_columns(df, "sharadar")]),
            "institutions": lambda df: iter([_csv_institutions(df)]),
            "insiders": lambda df: iter([_aggregate_insiders(df)]),
        }
        if kind not in readers:
            raise ValueError(f"알 수 없는 CSV 종류 '{kind}'. 지원: {sorted(readers)}")
        for chunk in pd.read_csv(path, chunksize=200_000):
            yield from readers[kind](chunk)

    # ------------------------------------------------------------------ 내부
    def _paginate(self, table: str, params: dict) -> Iterator[pd.DataFrame]:
        if self.api == "direct":
            yield from self._paginate_direct(table, params)
        else:
            yield from self._paginate_ndl(table, params)

    def _paginate_direct(self, table: str, params: dict) -> Iterator[pd.DataFrame]:
        """
        직판 API 윈도잉 페이지네이션.

        커서가 없으므로 날짜 컬럼 오름차순 정렬 + limit 만큼 받고,
        마지막 행의 날짜를 다음 요청의 `from` 으로 쓴다. 경계 날짜 행이
        중복 수신될 수 있으나 스토어 업서트(키 기반)가 멱등하게 걸러낸다.
        """
        slug = _DIRECT_TABLES.get(table, table.lower())
        url = _DIRECT_URL.format(table=slug)
        date_col = _DIRECT_PAGE_COL.get(table)
        window_from = params.pop("from", None)

        while True:
            query = {
                **params,
                "api_key": self.api_key,
                "format": "json",
                "limit": self.page_size,
            }
            if date_col:
                query["sort"] = f"{date_col}.asc"
                if window_from:
                    query["from"] = window_from
            frame = _parse_direct_payload(self._fetch(url, query))
            if frame.empty:
                return
            yield frame
            if date_col is None or len(frame) < self.page_size:
                return
            window_from = str(pd.to_datetime(frame[date_col]).max().date())

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


def _csv_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[df.get("dimension", DEFAULT_DIMENSION) == DEFAULT_DIMENSION]
    frame = normalize_columns(frame, "sharadar")
    validate_pit_frame(frame)
    return frame


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
    # transactionshares: 매수 양수 / 매도 음수 (Sharadar 규약)
    grouped = (
        df.groupby(["ticker", "calendardate"])
        .agg(insider_net_shares=("transactionshares", "sum"))
        .reset_index()
    )
    grouped["datekey"] = grouped["calendardate"] + pd.Timedelta(days=3)
    validate_pit_frame(grouped)
    return grouped
