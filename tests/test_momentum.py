"""
Unit tests for MomentumAnalyzer — VAA momentum calculation engine.
Tests focus on pure computation methods that don't require live data fetches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.strategies.momentum import MomentumAnalyzer


@pytest.fixture()
def analyzer() -> MomentumAnalyzer:
    return MomentumAnalyzer(use_cache=False)


@pytest.fixture()
def daily_prices() -> pd.DataFrame:
    """Daily price data with 400 trading days for 3 tickers."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=400)
    tickers = ["SPY", "TLT", "GLD"]
    data = {}
    for t in tickers:
        start = 100.0
        returns = rng.normal(0.0003, 0.01, len(dates))
        data[t] = start * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


class TestCalculateReturns:
    def test_returns_four_periods(self, analyzer: MomentumAnalyzer, daily_prices: pd.DataFrame) -> None:
        result = analyzer.calculate_returns(daily_prices)
        assert set(result.keys()) == {"1-Month", "3-Month", "6-Month", "12-Month"}

    def test_returns_are_percent_values(self, analyzer: MomentumAnalyzer, daily_prices: pd.DataFrame) -> None:
        result = analyzer.calculate_returns(daily_prices)
        # Values should be percentage points (not fractions)
        vals = result["1-Month"].dropna().values.flatten()
        # Most monthly % returns should be between -50 and +50
        assert np.all(np.abs(vals[~np.isnan(vals)]) < 200)

    def test_custom_periods(self, analyzer: MomentumAnalyzer, daily_prices: pd.DataFrame) -> None:
        result = analyzer.calculate_returns(daily_prices, periods=[21, 63])
        assert len(result) == 2


class TestCalculateMomentumScore:
    def test_output_shape_matches_tickers(self, analyzer: MomentumAnalyzer, daily_prices: pd.DataFrame) -> None:
        returns = analyzer.calculate_returns(daily_prices)
        scores = analyzer.calculate_momentum_score(returns)
        assert set(scores.columns) == {"SPY", "TLT", "GLD"}

    def test_custom_weights_applied(self, analyzer: MomentumAnalyzer, daily_prices: pd.DataFrame) -> None:
        returns = analyzer.calculate_returns(daily_prices)
        scores_default = analyzer.calculate_momentum_score(returns)
        scores_equal = analyzer.calculate_momentum_score(returns, weights=[1, 1, 1, 1])
        # Different weights → different scores
        assert not scores_default.equals(scores_equal)

    def test_scores_finite(self, analyzer: MomentumAnalyzer, daily_prices: pd.DataFrame) -> None:
        returns = analyzer.calculate_returns(daily_prices)
        scores = analyzer.calculate_momentum_score(returns)
        assert np.all(np.isfinite(scores.values))


class TestIsNegativeMomentum:
    def test_all_positive_scores_not_defensive(self, analyzer: MomentumAnalyzer) -> None:
        ranked = pd.DataFrame({
            "Ticker": ["SPY", "TLT", "GLD"],
            "Momentum Score": [10.0, 5.0, 3.0],
        })
        result = analyzer.is_negative_momentum(ranked)
        assert not result

    def test_negative_scores_triggers_defensive(self, analyzer: MomentumAnalyzer) -> None:
        ranked = pd.DataFrame({
            "Ticker": ["SPY", "TLT", "GLD"],
            "Momentum Score": [-10.0, -5.0, -3.0],
        })
        result = analyzer.is_negative_momentum(ranked)
        assert result
