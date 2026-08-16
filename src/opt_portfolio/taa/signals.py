"""모멘텀 시그널 — 순수 함수.

두 지표를 쓰는 이유가 다르다. **경보는 빠르게, 선택은 느리게** 라는 것이
BAA 논문의 표현이다.

- `momentum_13612w` : 카나리아(위험 경보) 판정용. 최근 1개월에 무게가 실린다
- `sma_ratio`       : 자산 선택용. 13개월 평균 대비라 훨씬 느리다

13612W 의 가중치 12/4/2/1 은 임의가 아니라 **연율화 계수**다 — 1개월 수익 ×12,
3개월 ×4, 6개월 ×2, 12개월 ×1 로 서로 다른 시간축의 연율 수익을 합한 값이다.
"""

from __future__ import annotations

import pandas as pd

#: (개월 수, 가중치) — Keller 13612W
_MOMENTUM_TERMS: tuple[tuple[int, int], ...] = ((1, 12), (3, 4), (6, 2), (12, 1))


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """일별 패널 → 월말 종가."""
    return daily.resample("ME").last()


def momentum_13612w(monthly: pd.DataFrame) -> pd.DataFrame:
    """13612W 모멘텀. 12개월 미만 구간은 NaN 이다."""
    score = None
    for months, weight in _MOMENTUM_TERMS:
        term = weight * monthly.pct_change(months)
        score = term if score is None else score + term
    assert score is not None  # _MOMENTUM_TERMS 가 비지 않음
    return score


def sma_ratio(monthly: pd.DataFrame, window: int = 13) -> pd.DataFrame:
    """현재가 / 직전 `window` 개월 평균. 1 보다 크면 상승 추세."""
    return monthly / monthly.rolling(window, min_periods=window).mean()
