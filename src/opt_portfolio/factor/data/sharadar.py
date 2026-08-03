"""
Sharadar 어댑터 (Nasdaq Data Link datatables API + 벌크 CSV)

두 가지 수급 경로를 지원한다:
- **API**: 증분 동기화 (`lastupdated >= since`). 일간 갱신용.
- **CSV**: 구독 시 제공되는 벌크 다운로드. 초기 적재용 (수 GB).

엔드포인트·컬럼명은 Nasdaq Data Link 문서 기준이며, sharadar.com 직판의
API 형태가 다르면 `_TABLE_URL` 과 컬럼 매핑만 고치면 된다 —
정규화 이후는 전부 벤더 중립이다.

환경변수: NASDAQ_DATA_LINK_API_KEY (또는 SHARADAR_API_KEY)
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

_TABLE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json"

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
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("NASDAQ_DATA_LINK_API_KEY")
            or os.environ.get("SHARADAR_API_KEY")
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
        cursor: str | None = None
        while True:
            query = {
                **params,
                "api_key": self.api_key,
                "qopts.per_page": self.page_size,
            }
            if cursor:
                query["qopts.cursor_id"] = cursor
            payload = self._fetch(_TABLE_URL.format(table=table), query)
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
    def _fetch(self, url: str, params: dict) -> dict:
        payload: dict = self._get_json(url, params)
        return payload


def _default_get_json(url: str, params: dict) -> dict:
    import requests

    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientAPIError(f"HTTP {resp.status_code}: {url}")
    resp.raise_for_status()
    payload: dict = resp.json()
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
