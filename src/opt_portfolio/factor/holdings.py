"""
현재 보유목록 산출 — 연구에서 운용으로 넘어가는 지점.

백테스트가 "과거에 무엇을 샀어야 했나"를 답한다면, 여기는 **"오늘 무엇을
사는가"**를 답한다. 두 답이 다른 규칙에서 나오면 그동안의 검증이 전부
무의미해지므로, 종목 선정(`_select`)과 비중(`compute_weights`)은 백테스트
엔진의 함수를 **그대로 재사용한다.**
"""

from __future__ import annotations

import pandas as pd

from opt_portfolio.factor.backtest.engine import MIN_COV_OBS, _select
from opt_portfolio.factor.portfolio.weights import compute_weights


def current_holdings(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    *,
    n_stocks: int = 20,
    weighting: str = "equal",
    max_weight: float = 0.10,
    universe: pd.DataFrame | None = None,
    market_caps: pd.Series | None = None,
    cov_window: int = 252,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    가장 최근(또는 지정) 신호일 기준 보유목록.

    Returns:
        ticker 인덱스, 컬럼 = weight / score / price. 비중 내림차순.
    """
    signal_date = pd.Timestamp(as_of) if as_of is not None else scores.index[-1]
    trade_date = close.index[close.index.searchsorted(signal_date, side="right") - 1]

    # 신호일이 거래일이 아니면(월말이 휴일 등) `_select` 가 한 달 전 스코어를
    # 집는다. 쓸 행을 명시적으로 고정하고 거래일 라벨을 붙여 그 사고를 막는다.
    score_date = scores.index[scores.index.searchsorted(signal_date, side="right") - 1]
    latest = scores.loc[[score_date]].set_axis(pd.DatetimeIndex([trade_date]))

    # 유니버스도 같은 이유로 정렬한다 — 인덱스가 어긋나면 `_select` 가 필터를
    # 조용히 건너뛴다. 있어야 할 필터가 사라지는 쪽이 더 위험하다.
    mask = None
    if universe is not None and len(universe):
        loc = universe.index.searchsorted(signal_date, side="right") - 1
        if loc >= 0:
            mask = universe.iloc[[loc]].set_axis(pd.DatetimeIndex([trade_date]))

    selected = _select(latest, close, mask, trade_date, n_stocks)
    if selected.empty:
        return pd.DataFrame(columns=["weight", "score", "price"])

    names = selected.index
    window = close.loc[:trade_date, names].tail(cov_window)
    returns = window.pct_change().dropna(how="all")
    if len(returns) < MIN_COV_OBS and weighting not in ("equal", "value"):
        # 공분산 추정이 불가능한 구간에서 조용히 이상한 비중을 내지 않는다
        weighting = "equal"

    weights = compute_weights(
        weighting,
        returns,
        selected,
        max_weight=max_weight,
        market_caps=market_caps,
    )
    out = pd.DataFrame(
        {
            "weight": weights.reindex(names),
            "score": selected,
            "price": close.loc[trade_date, names],
        }
    )
    return out.sort_values("weight", ascending=False)


def rebalance_plan(target: pd.Series, current: pd.Series) -> pd.DataFrame:
    """
    목표 비중 대 현재 비중 → 매매 계획.

    `diff` 는 목표−현재이며, 편도 회전율은 `diff.abs().sum()/2` 다
    (비용 모델의 회전율 규약과 같다).
    """
    names = target.index.union(current.index)
    tgt = target.reindex(names).fillna(0.0)
    cur = current.reindex(names).fillna(0.0)
    diff = tgt - cur

    action = pd.Series("유지", index=names, dtype=object)
    action[cur <= 0] = "매수"
    action[tgt <= 0] = "매도"

    plan = pd.DataFrame({"target": tgt, "current": cur, "diff": diff, "action": action})
    return plan.sort_values("diff", ascending=False)
