"""
Unit tests for RiskAnalyzer — comprehensive risk metric calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.analysis.risk import RiskAnalyzer


@pytest.fixture()
def analyzer() -> RiskAnalyzer:
    return RiskAnalyzer(risk_free_rate=0.02)


@pytest.fixture()
def positive_returns() -> pd.Series:
    """Consistent 1%/month returns — low risk, positive drift."""
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.01, 0.02, 60))


@pytest.fixture()
def negative_returns() -> pd.Series:
    """Consistent -1%/month returns."""
    rng = np.random.default_rng(2)
    return pd.Series(rng.normal(-0.01, 0.02, 60))


@pytest.fixture()
def price_series() -> pd.Series:
    """60-month price series derived from positive returns."""
    rng = np.random.default_rng(1)
    rets = rng.normal(0.01, 0.02, 60)
    return pd.Series(100 * np.cumprod(1 + rets))


class TestCalculateVolatility:
    def test_zero_volatility_constant_returns(self, analyzer: RiskAnalyzer) -> None:
        returns = pd.Series([0.01] * 12)
        vol = analyzer.calculate_volatility(returns)
        assert vol < 1e-10  # effectively zero

    def test_annualized_higher_than_period(self, analyzer: RiskAnalyzer, positive_returns: pd.Series) -> None:
        vol_ann = analyzer.calculate_volatility(positive_returns, annualize=True)
        vol_raw = analyzer.calculate_volatility(positive_returns, annualize=False)
        assert vol_ann > vol_raw

    def test_higher_vol_series(self, analyzer: RiskAnalyzer) -> None:
        low_vol = pd.Series([0.01] * 60 + [-0.01] * 60)
        high_vol = pd.Series([0.05, -0.05] * 60)
        assert analyzer.calculate_volatility(high_vol) > analyzer.calculate_volatility(low_vol)


class TestCalculateSharpeRatio:
    def test_positive_drift_positive_sharpe(self, analyzer: RiskAnalyzer, positive_returns: pd.Series) -> None:
        result = analyzer.calculate_sharpe_ratio(positive_returns)
        assert result > 0

    def test_negative_drift_negative_sharpe(self, analyzer: RiskAnalyzer, negative_returns: pd.Series) -> None:
        result = analyzer.calculate_sharpe_ratio(negative_returns)
        assert result < 0

    def test_zero_std_returns_zero(self, analyzer: RiskAnalyzer) -> None:
        result = analyzer.calculate_sharpe_ratio(pd.Series([0.0] * 12))
        assert result == 0.0


class TestCalculateMaxDrawdown:
    def test_monotone_growth_near_zero_mdd(self, analyzer: RiskAnalyzer, price_series: pd.Series) -> None:
        mdd, _peak, _trough = analyzer.calculate_max_drawdown(price_series)
        assert mdd >= 0.0

    def test_known_drawdown(self, analyzer: RiskAnalyzer) -> None:
        prices = pd.Series([100.0, 80.0, 90.0, 70.0, 110.0])
        mdd, _peak, _trough = analyzer.calculate_max_drawdown(prices)
        # Peak = 100, trough = 70 → 30% drawdown
        assert abs(mdd - 0.30) < 1e-6

    def test_flat_prices_zero_mdd(self, analyzer: RiskAnalyzer) -> None:
        prices = pd.Series([100.0] * 20)
        mdd, _peak, _trough = analyzer.calculate_max_drawdown(prices)
        assert mdd == 0.0


class TestCalculateVar:
    def test_var_is_positive(self, analyzer: RiskAnalyzer, positive_returns: pd.Series) -> None:
        result = analyzer.calculate_var(positive_returns)
        assert isinstance(result, float)

    def test_higher_confidence_higher_var(self, analyzer: RiskAnalyzer, positive_returns: pd.Series) -> None:
        var_95 = analyzer.calculate_var(positive_returns, confidence=0.95)
        var_99 = analyzer.calculate_var(positive_returns, confidence=0.99)
        assert var_99 >= var_95


class TestCalculateBeta:
    def test_same_series_beta_one(self, analyzer: RiskAnalyzer, positive_returns: pd.Series) -> None:
        beta = analyzer.calculate_beta(positive_returns, positive_returns)
        assert abs(beta - 1.0) < 1e-6

    def test_inverse_series_beta_negative(self, analyzer: RiskAnalyzer) -> None:
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        inverse = -returns
        beta = analyzer.calculate_beta(inverse, returns)
        assert beta < 0
