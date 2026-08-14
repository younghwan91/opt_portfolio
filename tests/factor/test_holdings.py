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


class TestPriceFieldSelection:
    """
    일별 패널은 필요한 필드만 싣는다.

    prices 테이블에는 10개 필드가 있지만 팩터·엔진이 실제로 쓰는 것은
    close·closeunadj·volume·mcap 넷뿐이다 (EV 팩터도 벤더 ev 대신
    mcap+debt-cashneq 로 직접 계산한다). 6,895종목 × 7,190일 패널 하나가
    약 400MB 이므로 안 쓰는 5개를 싣는 것만으로 2GB 가 낭비된다 —
    이 낭비가 OOM 을 세 번 냈다 (2026-08-14~15).
    """

    def _store(self, tmp_path):
        import numpy as np
        import pandas as pd

        from opt_portfolio.factor.data.store import PITStore

        store = PITStore(str(tmp_path / "t.duckdb"))
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        frame = pd.DataFrame(
            {
                "ticker": np.repeat(["A", "B"], len(idx)),
                "date": list(idx) * 2,
                "close": 100.0,
                "closeunadj": 100.0,
                "volume": 1e6,
                "mcap": 1e9,
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
            }
        )
        store.upsert_prices(frame)
        return store

    def test_loads_only_requested_fields(self, tmp_path) -> None:
        store = self._store(tmp_path)

        ctx = store.build_context(price_fields=("close", "volume"))

        assert set(ctx.daily) == {"close", "volume"}
        store.close()

    def test_default_loads_everything_present(self, tmp_path) -> None:
        """기본값은 기존 동작 그대로 — 명시하지 않으면 아무것도 안 바뀐다."""
        store = self._store(tmp_path)

        ctx = store.build_context()

        assert {"close", "closeunadj", "volume", "mcap", "open"} <= set(ctx.daily)
        store.close()

    def test_unknown_field_is_ignored_not_fabricated(self, tmp_path) -> None:
        store = self._store(tmp_path)

        ctx = store.build_context(price_fields=("close", "없는필드"))

        assert set(ctx.daily) == {"close"}
        store.close()
