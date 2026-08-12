"""
현재 보유목록 산출 — 연구에서 운용으로 넘어가는 지점.

가장 중요한 성질은 **백테스트와 같은 규칙으로 뽑히는가**다. 실전 목록이
다른 경로로 계산되면 그동안의 검증이 전부 무의미해진다. 그래서 엔진의
`_select` 와 `compute_weights` 를 그대로 재사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.holdings import current_holdings, rebalance_plan


@pytest.fixture
def scores_and_prices():
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    tickers = [f"T{i:02d}" for i in range(10)]
    rng = np.random.default_rng(5)
    scores = pd.DataFrame(rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers)
    # 마지막 신호일에 순위를 확정적으로 만든다
    scores.iloc[-1] = np.arange(len(tickers), dtype=float)
    cal = pd.date_range("2024-01-01", periods=200, freq="B")
    close = pd.DataFrame(100.0, index=cal, columns=tickers)
    return scores, close


class TestCurrentHoldings:
    def test_picks_top_n_by_score(self, scores_and_prices) -> None:
        scores, close = scores_and_prices

        held = current_holdings(scores, close, n_stocks=3)

        assert list(held.index) == ["T09", "T08", "T07"], "스코어 상위가 아니다"

    def test_weights_sum_to_one(self, scores_and_prices) -> None:
        scores, close = scores_and_prices

        held = current_holdings(scores, close, n_stocks=5)

        assert held["weight"].sum() == pytest.approx(1.0)

    def test_equal_weighting_is_equal(self, scores_and_prices) -> None:
        scores, close = scores_and_prices

        held = current_holdings(scores, close, n_stocks=4, weighting="equal")

        assert held["weight"].std() == pytest.approx(0.0, abs=1e-12)

    def test_universe_mask_excludes(self, scores_and_prices) -> None:
        """유니버스에서 빠진 종목은 스코어가 높아도 담기지 않는다."""
        scores, close = scores_and_prices
        mask = pd.DataFrame(True, index=scores.index, columns=scores.columns)
        mask.loc[:, "T09"] = False

        held = current_holdings(scores, close, n_stocks=3, universe=mask)

        assert "T09" not in held.index


class TestRebalancePlan:
    def test_reports_buy_sell_hold(self) -> None:
        target = pd.Series({"A": 0.5, "B": 0.5})
        current = pd.Series({"B": 0.5, "C": 0.5})

        plan = rebalance_plan(target, current)

        assert plan.loc["A", "action"] == "매수"
        assert plan.loc["C", "action"] == "매도"
        assert plan.loc["B", "action"] == "유지"

    def test_turnover_is_one_way(self) -> None:
        """회전율은 편도 기준 — 비용 모델과 규약을 맞춘다."""
        target = pd.Series({"A": 1.0})
        current = pd.Series({"B": 1.0})

        plan = rebalance_plan(target, current)

        assert plan["diff"].abs().sum() / 2 == pytest.approx(1.0)

    def test_empty_current_is_all_buy(self) -> None:
        plan = rebalance_plan(pd.Series({"A": 0.6, "B": 0.4}), pd.Series(dtype=float))

        assert set(plan["action"]) == {"매수"}
