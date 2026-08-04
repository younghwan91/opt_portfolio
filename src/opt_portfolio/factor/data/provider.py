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

    벤더가 같은 항목의 원시/USD 변형을 함께 줄 때 (debt & debtusd,
    cashneq & cashnequsd, close & closeadj) 리네임 타깃과 이름이 겹치는
    원시 컬럼은 먼저 버린다 — 안 버리면 컬럼명이 중복돼 스토어가 깨진다.
    USD 표준화 컬럼이 항상 우선이다. (실데이터 적재에서 발견, 2026-08-05)
    """
    rename = {
        spec.vendor[vendor]: spec.name
        for spec in FIELDS.values()
        if vendor in spec.vendor and spec.vendor[vendor] in df.columns
    }
    targets = set(rename.values())
    collisions = [c for c in df.columns if c in targets and c not in rename]
    return df.drop(columns=collisions).rename(columns=rename)


#: calendardate(표준화 달력 분기말)와 실제 결산일의 최대 허용 어긋남.
#: 비표준 결산월 기업(NKE 5월 결산 등)은 실제 분기말이 calendardate 보다
#: 최대 한 분기 가까이 이르므로, 공시일이 calendardate 직전에 올 수 있다.
_CALENDAR_SNAP_TOLERANCE = pd.Timedelta(days=92)


def validate_pit_frame(df: pd.DataFrame, *, date_col: str = "calendardate") -> None:
    """
    어댑터 출력의 PIT 계약 검증 — 스토어 진입 전 마지막 방어선.

    회계기간이 끝나기 전에 공시될 수는 없다. 다만 calendardate 는
    '표준화된 달력 분기말'이라 실제 결산일(reportperiod)과 다를 수 있다
    — 실데이터 검증에서 NKE(5월 결산)의 datekey 2025-12-30 <
    calendardate 2025-12-31 이 정상 케이스로 확인됨 (2026-08-05).

    - reportperiod 컬럼이 있으면: datekey >= reportperiod 를 엄격 검증
    - 없으면: calendardate 대비 92일(한 분기)까지의 조기 공시만 허용
    """
    for col in ("ticker", date_col, "datekey"):
        if col not in df.columns:
            raise ValueError(f"PIT 계약 위반: '{col}' 컬럼 누락")

    datekey = pd.to_datetime(df["datekey"])
    if "reportperiod" in df.columns:
        bad = datekey < pd.to_datetime(df["reportperiod"])
        label = "공시일 < 실제 결산일(reportperiod)"
        cols = ["ticker", "reportperiod", "datekey"]
    else:
        bad = datekey < pd.to_datetime(df[date_col]) - _CALENDAR_SNAP_TOLERANCE
        label = f"공시일이 {date_col} 보다 92일 이상 앞섬"
        cols = ["ticker", date_col, "datekey"]

    if bad.any():
        examples = df.loc[bad, cols].head(3)
        raise ValueError(f"PIT 계약 위반: {label} 행 {int(bad.sum())}개\n{examples}")
