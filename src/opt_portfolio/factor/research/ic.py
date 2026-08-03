"""
팩터 예측력 진단 — Rank IC, IC-IR, 감쇠 프로파일

퀀트 관점:
- Rank IC(Spearman)를 쓰는 이유: 팩터 값의 스케일·분포에 불변이고,
  극단값 하나가 Pearson IC 를 지배하는 것을 막는다.
- 순방향 수익률은 "신호 t 종가 → 진입 t+1 → 청산 t+1+h" 규약을 따른다.
  t 종가로 진입하는 규약은 신호와 체결이 동시라는 비현실적 가정이며,
  백테스트 엔진의 체결 규약과 반드시 일치해야 IC 가 실현 가능한 예측력을 재현한다.
- 일별 IC 는 자기상관이 심하다 (h일 보유면 h-1 일이 겹침). t-통계는
  겹침을 보정한 유효 표본수(n/h)로 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: 횡단면 IC 계산에 필요한 최소 종목 수 — 이보다 적으면 그 날짜는 NaN
MIN_CROSS_SECTION = 20


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    신호일 t 에 정렬된 순방향 수익률.

        fwd[t] = close[t+1+h] / close[t+1] − 1

    t+1 진입이므로 신호일 종가 정보는 수익률에 포함되지 않는다.
    """
    entry = close.shift(-1)
    exit_ = close.shift(-1 - horizon)
    return exit_ / entry - 1.0


def rank_ic(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """
    날짜별 Spearman Rank IC.

    두 패널에서 동시에 관측된 종목만으로 랭킹을 다시 매긴다 —
    한쪽만 NaN 인 종목을 남겨두면 랭크가 왜곡된다.
    """
    scores, fwd = scores.align(fwd, join="inner")
    valid = scores.notna() & fwd.notna()

    x = scores.where(valid).rank(axis=1)
    y = fwd.where(valid).rank(axis=1)

    xm = x.sub(x.mean(axis=1), axis=0)
    ym = y.sub(y.mean(axis=1), axis=0)
    cov = (xm * ym).sum(axis=1)
    den = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1))

    ic = cov / den.where(den > 0)
    return ic.where(valid.sum(axis=1) >= MIN_CROSS_SECTION)


@dataclass(frozen=True)
class ICSummary:
    """팩터 하나의 IC 프로파일 요약."""

    mean: float
    std: float
    ir: float          # IC-IR = mean / std
    t_stat: float      # 겹침 보정 유효 표본 기준
    hit_rate: float    # IC > 0 인 날짜 비율
    n_obs: int

    def is_significant(self, threshold: float = 2.0) -> bool:
        return abs(self.t_stat) >= threshold


def summarize_ic(ic: pd.Series, horizon: int = 21) -> ICSummary:
    """
    IC 시계열 요약.

    Args:
        horizon: 보유기간(일). 일별 IC 는 h-1 일이 겹치므로
            유효 표본수를 n/h 로 줄여 t-통계를 보수적으로 계산한다.
    """
    clean = ic.dropna()
    n = len(clean)
    if n < 2:
        return ICSummary(np.nan, np.nan, np.nan, np.nan, np.nan, n)

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    n_eff = max(n / max(horizon, 1), 2.0)
    t_stat = mean / std * np.sqrt(n_eff) if std > 0 else np.nan
    ir = mean / std if std > 0 else np.nan
    return ICSummary(
        mean=mean,
        std=std,
        ir=ir,
        t_stat=float(t_stat),
        hit_rate=float((clean > 0).mean()),
        n_obs=n,
    )


def ic_decay(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 21, 63, 126),
) -> pd.Series:
    """
    보유기간별 평균 IC — 팩터 신호의 반감기.

    밸류 팩터는 느리게 감쇠하고(수개월), 단기 반전은 며칠 만에 죽는다.
    리밸런싱 주기는 IC 감쇠가 절반으로 떨어지기 전으로 잡아야
    신호가 살아있는 동안 포지션이 잡힌다.
    """
    return pd.Series(
        {h: rank_ic(scores, forward_returns(close, h)).mean() for h in horizons},
        name="mean_ic",
    )
