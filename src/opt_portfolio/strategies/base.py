"""
Abstract Strategy Base Class

Defines the common interface for all portfolio strategies in opt_portfolio.
Concrete strategies should inherit from AbstractStrategy and implement
the abstract methods to ensure consistent behavior across the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class AbstractStrategy(ABC):
    """
    Abstract base class for all quantitative portfolio strategies.

    All strategies must implement ``calculate()``, ``rank()``, and ``rebalance()``
    to guarantee a uniform interface for the BacktestEngine.
    """

    @abstractmethod
    def calculate(self, price_data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Compute the strategy's core signal or score for each asset.

        Args:
            price_data: Historical price DataFrame (dates × tickers).
            **kwargs: Strategy-specific parameters.

        Returns:
            DataFrame of scores/signals with the same column structure as the input.
        """

    @abstractmethod
    def rank(self, scores: pd.DataFrame) -> pd.Series:
        """
        Rank assets based on computed scores.

        Args:
            scores: Output from ``calculate()``.

        Returns:
            Series mapping ticker → rank (1 = best).
        """

    @abstractmethod
    def rebalance(
        self,
        ranks: pd.Series,
        current_weights: pd.Series,
        **kwargs: Any,
    ) -> pd.Series:
        """
        Determine target portfolio weights from asset ranks.

        Args:
            ranks: Output from ``rank()``.
            current_weights: Current portfolio weights (ticker → weight).
            **kwargs: Strategy-specific parameters (e.g., top_n, threshold).

        Returns:
            Series of target weights that sum to 1.0.
        """
