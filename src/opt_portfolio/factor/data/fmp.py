"""
FMP 어댑터 — 보조 데이터 전용 (공매도 잔고, 애널리스트 추정치)

가격·재무의 1차 소스는 Sharadar 다. FMP 는 Sharadar 에 없는 두 가지만 맡는다:
- 공매도 잔고 (SHORT_INT_CHG 팩터) — FINRA 격주 데이터
- 선행 EPS 성장률 (PEG_FWD 팩터) — 애널리스트 컨센서스

환경변수: FMP_API_KEY
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com"


class TransientAPIError(RuntimeError):
    """재시도 대상 (429 / 5xx)."""


class FMPProvider:
    """
    Args:
        get_json: HTTP GET 주입 지점 — 테스트에서 가짜 응답으로 대체.
    """

    name = "fmp"

    def __init__(
        self,
        api_key: str | None = None,
        get_json: Callable[[str, dict], list | dict] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY", "")
        self._get_json = get_json or _default_get_json

    def short_interest(self, tickers: list[str]) -> Iterator[pd.DataFrame]:
        """
        종목별 공매도 잔고 시계열 → prices 테이블 병합용 (ticker, date, short_interest).

        FINRA 결제일 기준 격주 발표 + 약 8일 공표 지연이 있으나, 스토어는
        일별 그리드에 저장하고 발표일(date 컬럼) 기준으로만 노출하므로
        지연이 자연 반영된다.
        """
        for ticker in tickers:
            payload = self._fetch(
                f"{_BASE}/api/v4/short-interest",
                {"symbol": ticker, "apikey": self.api_key},
            )
            if not payload:
                continue
            df = pd.DataFrame(payload)
            if df.empty or "date" not in df.columns:
                continue
            yield pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": pd.to_datetime(df["date"]),
                    "short_interest": pd.to_numeric(df.get("shortInterest"), errors="coerce"),
                }
            )

    def analyst_estimates(self, tickers: list[str]) -> Iterator[pd.DataFrame]:
        """
        분기 EPS 컨센서스 → estimates 테이블 (ticker, calendardate, datekey, eps_growth_fwd).

        선행 성장률 = (차기 분기 추정 EPS / 최근 실적 EPS) − 1.
        datekey 는 조회 시점 스냅샷 날짜 — FMP 는 추정치 이력의 PIT 를
        제공하지 않으므로, **증분 수집을 시작한 날짜부터만** PIT 가 성립한다.
        과거로 소급한 추정치 백필은 구조적으로 look-ahead 이므로 지원하지 않는다.
        """
        snapshot = pd.Timestamp.now().normalize()
        for ticker in tickers:
            payload = self._fetch(
                f"{_BASE}/api/v3/analyst-estimates/{ticker}",
                {"period": "quarter", "apikey": self.api_key},
            )
            if not payload:
                continue
            df = pd.DataFrame(payload)
            if df.empty or "date" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            eps = pd.to_numeric(df.get("estimatedEpsAvg"), errors="coerce")
            growth = eps.pct_change(fill_method=None)
            yield pd.DataFrame(
                {
                    "ticker": ticker,
                    "calendardate": df["date"],
                    "datekey": snapshot,
                    "eps_growth_fwd": growth.shift(-1),  # t 시점의 '차기' 성장률
                    "eps_est_fwd": eps.shift(-1),
                }
            ).dropna(subset=["eps_growth_fwd"])

    @retry(
        retry=retry_if_exception_type(TransientAPIError),
        wait=wait_exponential(multiplier=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _fetch(self, url: str, params: dict) -> list | dict:
        payload: list | dict = self._get_json(url, params)
        return payload


def _default_get_json(url: str, params: dict) -> list | dict:
    import requests

    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientAPIError(f"HTTP {resp.status_code}: {url}")
    resp.raise_for_status()
    payload: list | dict = resp.json()
    return payload
