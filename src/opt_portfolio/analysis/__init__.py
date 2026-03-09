"""Analysis module initialization."""

from .backtest import BacktestEngine, BacktestResult
from .data_fetcher import DataFetcher
from .metrics import calculate_cagr, calculate_max_drawdown, calculate_sharpe_ratio
from .optimizer import OptimizationResult, PortfolioOptimizer
from .performance import PerformanceAnalyzer
from .report import BacktestReporter
from .risk import RiskAnalyzer

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestReporter",
    "DataFetcher",
    "RiskAnalyzer",
    "PerformanceAnalyzer",
    "PortfolioOptimizer",
    "OptimizationResult",
    "calculate_cagr",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
]
