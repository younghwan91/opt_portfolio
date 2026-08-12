"""
시장 레짐 분류 — "지금과 비슷했던 때 무엇이 통했나"를 묻기 위한 상태 정의.

퀀트 관점:
- 레짐을 HMM 같은 잠재변수 모형으로 추정할 수도 있지만, 여기서는 **관측
  가능한 두 축**으로 나눈다: 추세(벤치마크 12개월 수익 부호)와 변동성
  (실현변동성의 과거 대비 위치). 이유는 강건성이다 — 잠재 레짐은 추정
  오차와 라벨 스위칭이 있고, 그 오차가 팩터 선택 오차와 곱해진다.
  DeMiguel et al.(2009) 이후 '1/N 을 이기기'가 최적화의 기준선인 것과
  같은 논리로, 레짐도 단순한 쪽이 먼저다.
- **모든 판정은 그 시점까지의 데이터만 쓴다.** 변동성 임계값을 전체 표본의
  중앙값으로 잡으면 미래를 보는 것이므로, 확장 윈도 분위수를 쓴다.
- 레짐은 예측이 아니라 **조건부 학습의 조건**이다. "지금이 어떤 상태인가"만
  말하고 "다음에 무엇이 온다"는 말하지 않는다.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

#: 추세 판정 기간 (거래일). 12개월 — 모멘텀 문헌의 표준.
TREND_DAYS = 252

#: 변동성 측정 기간 (거래일). 3개월 — 레짐 전환에 반응하되 노이즈는 걸러낸다.
VOL_DAYS = 63

#: 변동성 '높음' 판정 분위수. 확장 윈도로 계산해 look-ahead 를 막는다.
VOL_QUANTILE = 0.6

#: 분위수 추정에 필요한 최소 관측 — 이보다 짧으면 레짐을 말하지 않는다.
MIN_HISTORY = 504


class Regime(str, Enum):
    """추세 × 변동성 2×2. 값은 로그·설정에 그대로 쓰이는 문자열이다."""

    BULL_CALM = "bull_calm"
    BULL_TURBULENT = "bull_turbulent"
    BEAR_CALM = "bear_calm"
    BEAR_TURBULENT = "bear_turbulent"
    UNKNOWN = "unknown"  # 히스토리 부족


def classify(benchmark_close: pd.Series) -> pd.Series:
    """
    벤치마크 종가 → 일자별 레짐.

    Args:
        benchmark_close: 벤치마크(예: SPY) 종가. 결측은 앞값으로 채운다.

    Returns:
        같은 인덱스의 `Regime` 값 시리즈. 히스토리가 부족한 앞 구간은 UNKNOWN.
    """
    close = benchmark_close.dropna().astype(float)
    if close.empty:
        return pd.Series(dtype=object)

    trend_up = close.pct_change(TREND_DAYS) > 0
    realized_vol = close.pct_change().rolling(VOL_DAYS).std() * np.sqrt(252)

    # 확장 윈도 분위수 — 그 시점까지의 분포만 본다 (look-ahead 차단)
    threshold = realized_vol.expanding(min_periods=MIN_HISTORY).quantile(VOL_QUANTILE)
    turbulent = realized_vol > threshold

    out = pd.Series(Regime.UNKNOWN.value, index=close.index, dtype=object)
    known = threshold.notna() & realized_vol.notna() & close.pct_change(TREND_DAYS).notna()
    out[known & trend_up & ~turbulent] = Regime.BULL_CALM.value
    out[known & trend_up & turbulent] = Regime.BULL_TURBULENT.value
    out[known & ~trend_up & ~turbulent] = Regime.BEAR_CALM.value
    out[known & ~trend_up & turbulent] = Regime.BEAR_TURBULENT.value
    return out


def factor_ic_by_regime(
    panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    레짐별 팩터 Rank IC 평균.

    "지금과 비슷했던 때 무엇이 통했나"의 답을 만드는 표다.
    행=팩터, 열=레짐.
    """
    from opt_portfolio.factor.research.ic import rank_ic

    dates = forward_returns.index
    aligned = regimes.reindex(dates, method="ffill")
    rows: dict[str, dict[str, float]] = {}
    for name, panel in panels.items():
        ic = rank_ic(panel.reindex(dates), forward_returns)
        rows[name] = {
            str(reg): float(ic[aligned == reg].mean())
            for reg in aligned.dropna().unique()
            if (aligned == reg).sum() >= 12  # 관측 1년 미만 레짐은 신뢰하지 않는다
        }
    return pd.DataFrame(rows).T.sort_index()


def weights_for_regime(
    ic_table: pd.DataFrame,
    regime: str,
    *,
    floor: float = 0.0,
) -> dict[str, float]:
    """
    레짐별 IC → 팩터 가중치.

    IC 가 양(+)인 팩터만 남기고 IC 에 비례해 배분한다. 음(−)인 팩터를
    부호 뒤집어 쓰지 않는 이유는, 그 반전이 표본 특유의 잡음일 가능성이
    크기 때문이다 — 뒤집어 쓰려면 사전 근거가 따로 있어야 한다.

    해당 레짐에서 양의 IC 를 가진 팩터가 없으면 **균등 가중으로 후퇴**한다.
    조건부 판단이 근거를 잃었을 때 돌아갈 곳은 1/N 이다.
    """
    if regime not in ic_table.columns:
        return {}
    ic = ic_table[regime].dropna()
    positive = ic[ic > floor]
    if positive.empty:
        return dict.fromkeys(ic_table.index, 1.0)
    return {name: float(v) for name, v in (positive / positive.sum()).items()}
