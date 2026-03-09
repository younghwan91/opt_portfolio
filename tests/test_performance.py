"""
Unit tests for PerformanceAnalyzer — rolling returns, statistics, and drawdown analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.analysis.performance import PerformanceAnalyzer


@pytest.fixture()
def perf() -> PerformanceAnalyzer:
    return PerformanceAnalyzer()


@pytest.fixture()
def monthly_returns() -> pd.Series:
    """48 monthly returns with positive drift."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    return pd.Series(rng.normal(0.008, 0.03, 48), index=dates)


@pytest.fixture()
def equity_curve(monthly_returns: pd.Series) -> pd.Series:
    return (1 + monthly_returns).cumprod() * 100


class TestCalculateCagr:
    def test_known_10pct_cagr(self, perf: PerformanceAnalyzer) -> None:
        result = perf.calculate_cagr(100.0, 121.0, years=2)
        assert abs(result - 0.10) < 1e-6

    def test_zero_years_returns_zero(self, perf: PerformanceAnalyzer) -> None:
        result = perf.calculate_cagr(100.0, 200.0, years=0)
        assert result == 0.0

    def test_flat_growth_returns_zero(self, perf: PerformanceAnalyzer) -> None:
        result = perf.calculate_cagr(100.0, 100.0, years=3)
        assert abs(result) < 1e-9


class TestCalculateRollingReturns:
    def test_output_length_correct(self, perf: PerformanceAnalyzer, monthly_returns: pd.Series) -> None:
        result = perf.calculate_rolling_returns(monthly_returns, window=12)
        # Rolling window of 12 on 48 → 37 valid entries
        assert len(result.dropna()) == len(monthly_returns) - 12 + 1

    def test_window_1_equals_input(self, perf: PerformanceAnalyzer, monthly_returns: pd.Series) -> None:
        result = perf.calculate_rolling_returns(monthly_returns, window=1)
        pd.testing.assert_series_equal(result.dropna(), monthly_returns, check_names=False)


class TestCalculateMonthlyStatistics:
    def test_returns_expected_keys(self, perf: PerformanceAnalyzer, monthly_returns: pd.Series) -> None:
        stats = perf.calculate_monthly_statistics(monthly_returns)
        for key in ("mean", "std", "win_rate"):
            assert key in stats

    def test_win_rate_between_0_and_1(self, perf: PerformanceAnalyzer, monthly_returns: pd.Series) -> None:
        stats = perf.calculate_monthly_statistics(monthly_returns)
        assert 0.0 <= stats["win_rate"] <= 1.0

    def test_all_positive_win_rate_is_1(self, perf: PerformanceAnalyzer) -> None:
        returns = pd.Series([0.01] * 12)
        stats = perf.calculate_monthly_statistics(returns)
        assert stats["win_rate"] == 1.0


class TestAnalyzeByYear:
    def test_returns_dataframe(self, perf: PerformanceAnalyzer, monthly_returns: pd.Series) -> None:
        result = perf.analyze_by_year(monthly_returns)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_year_column_present(self, perf: PerformanceAnalyzer, monthly_returns: pd.Series) -> None:
        result = perf.analyze_by_year(monthly_returns)
        assert "Year" in result.columns

    def test_correct_years(self, perf: PerformanceAnalyzer, monthly_returns: pd.Series) -> None:
        result = perf.analyze_by_year(monthly_returns)
        years = set(result["Year"].astype(int))
        assert years == {2020, 2021, 2022, 2023}
