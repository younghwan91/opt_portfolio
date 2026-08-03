"""
데이터 프로바이더 프로토콜 — 벤더 중립 계약

어댑터의 유일한 책임: 벤더 API/파일 → **정규화 프레임** 변환.
정규화 프레임의 컬럼은 schema.FIELDS 의 표준 이름이며,
PIT 키 (ticker, calendardate, datekey) 를 반드시 포함한다.

팩터·스토어·파이프라인은 벤더를 모른다 — 벤더 교체 시 어댑터만 바꾼다.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import pandas as pd

from opt_portfolio.factor.data.schema import FIELDS


@runtime_checkable
class DataProvider(Protocol):
    """모든 어댑터가 구현하는 계약."""

    name: str

    def fundamentals(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        """분기 재무 청크 스트림 (정규화 컬럼 + ticker/calendardate/datekey)."""
        ...

    def prices(self, since: str | None = None) -> Iterator[pd.DataFrame]:
        """일별 가격 청크 스트림 (정규화 컬럼 + ticker/date)."""
        ...

    def tickers(self) -> pd.DataFrame:
        """티커 메타 (ticker + sector/industry/location/category/name/...)."""
        ...


def normalize_columns(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """
    벤더 컬럼명 → 표준 필드명.

    스키마에 매핑이 등록된 컬럼만 이름을 바꾸고, PIT 키 컬럼은 그대로 둔다.
    매핑되지 않은 벤더 컬럼은 보존된다 (스토어가 무시).
    """
    rename = {spec.vendor[vendor]: spec.name for spec in FIELDS.values() if vendor in spec.vendor}
    return df.rename(columns=rename)


def validate_pit_frame(df: pd.DataFrame, *, date_col: str = "calendardate") -> None:
    """
    어댑터 출력의 PIT 계약 검증 — 스토어 진입 전 마지막 방어선.

    datekey < calendardate 인 행은 명백한 데이터 오류다
    (회계기간이 끝나기 전에 공시될 수 없다).
    """
    for col in ("ticker", date_col, "datekey"):
        if col not in df.columns:
            raise ValueError(f"PIT 계약 위반: '{col}' 컬럼 누락")
    bad = pd.to_datetime(df["datekey"]) < pd.to_datetime(df[date_col])
    if bad.any():
        examples = df.loc[bad, ["ticker", date_col, "datekey"]].head(3)
        raise ValueError(f"PIT 계약 위반: 공시일 < 회계기간말 행 {int(bad.sum())}개\n{examples}")
