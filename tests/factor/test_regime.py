"""
레짐 분류 — "지금과 비슷했던 때 무엇이 통했나"의 조건 정의.

가장 중요한 성질은 **look-ahead 차단**이다. 변동성 임계값을 전체 표본으로
잡으면 2008년의 레짐 판정에 2020년 데이터가 섞인다. 그러면 레짐 조건부
백테스트 전체가 조용히 미래를 본다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.research.regime import (
    MIN_HISTORY,
    Regime,
    classify,
    weights_for_regime,
)


def _series(returns: np.ndarray, start: str = "2000-01-03") -> pd.Series:
    idx = pd.date_range(start, periods=len(returns), freq="B")
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)


class TestClassify:
    def test_rising_calm_market_is_bull_calm(self) -> None:
        rng = np.random.default_rng(1)
        close = _series(rng.normal(0.0006, 0.004, 1500))  # 우상향·저변동

        regimes = classify(close)

        assert regimes.iloc[-1] == Regime.BULL_CALM.value

    def test_falling_market_is_bear(self) -> None:
        rng = np.random.default_rng(2)
        close = _series(rng.normal(-0.0008, 0.010, 1500))

        regimes = classify(close)

        assert regimes.iloc[-1].startswith("bear")

    def test_short_history_is_unknown(self) -> None:
        """히스토리가 부족하면 레짐을 말하지 않는다 — 추측보다 모른다가 낫다."""
        rng = np.random.default_rng(3)
        close = _series(rng.normal(0.0005, 0.01, MIN_HISTORY - 50))

        assert (classify(close) == Regime.UNKNOWN.value).all()

    def test_threshold_uses_only_past_data(self) -> None:
        """
        핵심 불변식. 뒤에 변동성 폭발 구간을 **덧붙여도** 앞 구간의 레짐
        판정이 바뀌면 안 된다. 바뀐다면 임계값이 미래를 본 것이다.
        """
        rng = np.random.default_rng(4)
        base = rng.normal(0.0005, 0.006, 2000)
        calm_only = _series(base)
        with_future_crash = _series(np.concatenate([base, rng.normal(-0.003, 0.05, 400)]))

        before = classify(calm_only)
        after = classify(with_future_crash).reindex(before.index)

        assert (before == after).all(), "미래 데이터가 과거 레짐 판정을 바꿨다"


class TestRegimeWeights:
    def _table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bull_calm": {"MOM": 0.04, "VALUE": -0.01, "QUALITY": 0.02},
                "bear_turbulent": {"MOM": -0.03, "VALUE": 0.01, "QUALITY": 0.03},
            }
        )

    def test_positive_ic_factors_get_weight_in_proportion(self) -> None:
        w = weights_for_regime(self._table(), "bull_calm")

        assert set(w) == {"MOM", "QUALITY"}, "IC 가 음수인 팩터가 섞였다"
        assert w["MOM"] > w["QUALITY"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_regime_changes_the_factor_set(self) -> None:
        """이 기능의 존재 이유 — 레짐이 바뀌면 쓰는 팩터가 바뀐다."""
        table = self._table()

        assert set(weights_for_regime(table, "bull_calm")) != set(
            weights_for_regime(table, "bear_turbulent")
        )

    def test_falls_back_to_equal_when_nothing_works(self) -> None:
        """양의 IC 가 하나도 없으면 1/N 으로 돌아간다."""
        table = pd.DataFrame({"bear_calm": {"A": -0.02, "B": -0.01}})

        w = weights_for_regime(table, "bear_calm")

        assert set(w) == {"A", "B"}
        assert w["A"] == w["B"]

    def test_unknown_regime_yields_no_weights(self) -> None:
        assert weights_for_regime(self._table(), "unknown") == {}
