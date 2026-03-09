"""
Optimal Portfolio Management System

A professional-grade portfolio management system implementing tactical asset allocation
strategies including VAA (Vigilant Asset Allocation) with advanced quantitative analytics.

Features:
- VAA momentum-based ETF selection
- Ornstein-Uhlenbeck process forecasting
- Smart DuckDB caching for market data
- Automated portfolio rebalancing
- Risk metrics and performance analytics
"""

import logging

__version__ = "1.0.0"
__author__ = "Portfolio Management Team"

from .analysis.backtest import BacktestEngine
from .analysis.risk import RiskAnalyzer
from .core.cache import DataCache, get_cache
from .core.portfolio import Portfolio
from .strategies.momentum import MomentumAnalyzer
from .strategies.vaa import VAAStrategy

# Configure a NullHandler so library users can set up their own logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DataCache",
    "get_cache",
    "Portfolio",
    "VAAStrategy",
    "MomentumAnalyzer",
    "BacktestEngine",
    "RiskAnalyzer",
]
