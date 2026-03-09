"""
Shared Financial Metrics Module

Provides standalone utility functions for common financial calculations
(CAGR, Sharpe Ratio, Max Drawdown) used across multiple modules.
Centralizing these prevents duplication and ensures consistent formulas.

퀀트 관점:
- CAGR = (최종가치/초기가치)^(1/년수) - 1
- Sharpe = (평균수익 - 무위험수익) / 수익 표준편차 × √252(연환산)
- Max Drawdown = 고점 대비 최대 낙폭 (포트폴리오 위험의 핵심 지표)
"""

import numpy as np
import pandas as pd


def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Investment period in years

    Returns:
        CAGR as a decimal (e.g. 0.12 for 12%)
    """
    if years <= 0 or initial_value <= 0:
        return 0.0
    return (final_value / initial_value) ** (1.0 / years) - 1


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 12,
) -> float:
    """
    Calculate annualised Sharpe Ratio.

    Args:
        returns: Periodic return series (e.g. monthly)
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Sharpe ratio (annualised)
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    periodic_rf = risk_free_rate / periods_per_year
    excess = returns - periodic_rf
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate Maximum Drawdown from an equity curve.

    Args:
        equity_curve: Portfolio value over time

    Returns:
        Max drawdown as a positive decimal (e.g. 0.20 for 20% drawdown)
    """
    if equity_curve.empty:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(abs(drawdown.min()))
