"""월별 리밸런싱 엔진.

**신호는 t월말 종가로 정하고 수익은 t+1월에 얻는다.** 같은 달 수익을 쓰면
룩어헤드이고, 그건 이 저장소가 구조로 막기로 한 실패 유형이다.

비용은 **회전한 만큼만** 문다. 비중이 안 바뀌면 0 이다 — 매달 물리면 정적
배분 기준선이 부당하게 불리해져 비교가 망가진다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import momentum_13612w, sma_ratio, to_monthly
from .strategy import StrategySpec, is_defensive, select_weights


@dataclass(frozen=True)
class BacktestOutput:
    returns: pd.Series
    equity: pd.Series
    selections: pd.Series
    defensive_ratio: float


def run_backtest(
    spec: StrategySpec,
    daily: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    cost_bps: float = 10.0,
) -> BacktestOutput:
    """월별 리밸런싱 백테스트.

    Args:
        spec: 전략 선언
        daily: 일별 배당조정 가격 패널
        start/end: 검증 구간 (None 이면 데이터 전체)
        cost_bps: 편도 거래비용 (bp). 회전한 비중에만 적용된다.
    """
    monthly = to_monthly(daily)
    mom = momentum_13612w(monthly)
    sel = sma_ratio(monthly, window=13)
    fwd = monthly.pct_change().shift(-1)

    needed = spec.tickers()
    usable = mom.dropna(how="any", subset=needed).index
    if start is not None:
        usable = usable[usable >= start]
    if end is not None:
        usable = usable[usable <= end]
    if len(usable) == 0:
        raise ValueError(f"[{spec.name}] 평가 가능한 시점이 없다 — 데이터 구간을 확인하라")

    month_index = monthly.index
    prev: dict[str, float] = {}
    dates, rets, picks, defensive_flags = [], [], [], []
    entry_date: pd.Timestamp | None = None  # 자본 곡선의 시작점 — 첫 회전 직전 시점

    for date in usable:
        pos = month_index.get_loc(date)
        if pos + 1 >= len(month_index):
            continue  # 다음 달이 없는 마지막 시점 — 수익을 매길 대상이 없다
        realization_date = month_index[pos + 1]

        nxt = fwd.loc[date]
        weights = select_weights(spec, mom, sel, date)
        gross = float(sum(w * nxt[t] for t, w in weights.items()))
        if not np.isfinite(gross):
            continue  # 다음 달 가격이 없는 마지막 시점

        all_tickers = set(weights) | set(prev)
        turnover = sum(abs(weights.get(t, 0.0) - prev.get(t, 0.0)) for t in all_tickers)
        cost = turnover * cost_bps / 10_000.0

        if entry_date is None:
            entry_date = date
        dates.append(realization_date)
        rets.append(gross - cost)
        picks.append(",".join(sorted(weights)))
        defensive_flags.append(is_defensive(spec, mom, date))
        prev = weights

    returns = pd.Series(rets, index=pd.DatetimeIndex(dates), name=spec.name)
    growth = (1 + returns).cumprod() * 10_000.0
    if entry_date is not None:
        base = pd.Series([10_000.0], index=pd.DatetimeIndex([entry_date]))
        equity = pd.concat([base, growth])
    else:
        equity = growth
    return BacktestOutput(
        returns=returns,
        equity=equity,
        selections=pd.Series(picks, index=returns.index),
        defensive_ratio=float(np.mean(defensive_flags)) if defensive_flags else 0.0,
    )
