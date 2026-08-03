"""
분위수 분석 — 스프레드, 단조성, 팩터 회전율

퀀트 관점:
- IC 는 선형 예측력만 본다. 분위수 분석은 "팩터가 상위에서만 작동하는가,
  하위에서만 작동하는가"를 드러낸다 — 롱온리 전략이면 상위 분위의
  초과수익이 전부이고, 하위 분위의 저성과는 실현할 수 없는 신호다.
- 단조성(monotonicity)이 깨진 팩터는 분위 경계 파라미터에 과적합하기 쉽다.
- 회전율은 팩터의 '실현 비용'이다. IC 가 같아도 회전율이 3배면
  비용 차감 후 알파는 전혀 다르다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from opt_portfolio.factor.research.ic import MIN_CROSS_SECTION


def assign_quantiles(scores: pd.DataFrame, n_quantiles: int = 10) -> pd.DataFrame:
    """
    날짜별 횡단면 분위수 배정 (1 = 최하위, n = 최상위).

    랭크 백분위를 등간격으로 자른다 — pd.qcut 의 중복 경계 문제를 피하고
    모든 날짜에서 분위 정의가 동일하게 유지된다.
    """
    pct = scores.rank(axis=1, pct=True)
    q = np.ceil(pct * n_quantiles)
    return q.where(scores.notna())


def quantile_returns(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """분위별 동일가중 평균 순방향 수익률 — (날짜 × 분위) 프레임."""
    scores, fwd = scores.align(fwd, join="inner")
    q = assign_quantiles(scores.where(fwd.notna()), n_quantiles)

    out = {}
    for k in range(1, n_quantiles + 1):
        mask = q.eq(k)
        row_count = mask.sum(axis=1)
        out[k] = fwd.where(mask).mean(axis=1).where(
            row_count >= MIN_CROSS_SECTION // n_quantiles
        )
    return pd.DataFrame(out)


@dataclass(frozen=True)
class QuantileReport:
    """분위수 분석 종합."""

    mean_by_quantile: pd.Series   # 분위별 시간평균 수익률
    spread: float                  # 최상위 − 최하위 (기간당)
    spread_t: float                # 스프레드 t-통계
    monotonicity: float            # 분위 순서 vs 평균수익 Spearman (−1..1)
    top_turnover: float            # 최상위 분위 구성 회전율 (기간당)


def analyze_quantiles(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    n_quantiles: int = 10,
) -> QuantileReport:
    qret = quantile_returns(scores, fwd, n_quantiles)
    mean_by_q = qret.mean()

    spread_series = (qret[n_quantiles] - qret[1]).dropna()
    n = len(spread_series)
    spread = float(spread_series.mean()) if n else np.nan
    spread_t = (
        float(spread / spread_series.std(ddof=1) * np.sqrt(n))
        if n > 1 and spread_series.std(ddof=1) > 0
        else np.nan
    )

    ranks = pd.Series(range(1, n_quantiles + 1), index=mean_by_q.index, dtype=float)
    mono = float(ranks.corr(mean_by_q, method="spearman"))

    return QuantileReport(
        mean_by_quantile=mean_by_q,
        spread=spread,
        spread_t=spread_t,
        monotonicity=mono,
        top_turnover=top_quantile_turnover(scores, n_quantiles),
    )


def top_quantile_turnover(
    scores: pd.DataFrame,
    n_quantiles: int = 10,
    freq: str = "ME",
) -> float:
    """
    최상위 분위 구성종목의 기간당 평균 교체율.

        turnover = 1 − |A_t ∩ A_{t−1}| / |A_{t−1}|

    월말 스냅샷 기준. 0.5 면 매달 절반이 교체된다는 뜻 —
    비용 모델과 곱해 팩터의 실현 가능 알파를 추정하는 입력이 된다.
    """
    q = assign_quantiles(scores, n_quantiles)
    snapshots = q.resample(freq).last()

    prev: set[str] | None = None
    rates = []
    for _, row in snapshots.iterrows():
        members = set(row.index[row.eq(n_quantiles)])
        if prev and members:
            rates.append(1.0 - len(members & prev) / len(prev))
        if members:
            prev = members
    return float(np.mean(rates)) if rates else np.nan
