"""
Data Fetching Module for Backtest Engine

Encapsulates all market data retrieval logic, separating data access
from the backtest computation logic.

퀀트 관점:
- 데이터 레이어 분리로 테스트 가능성 향상
- 캐시 우선 전략으로 API 호출 최소화
"""

import logging

import pandas as pd

from ..config import ASSETS
from ..core.cache import DataCache, get_cache
from ..strategies.momentum import MomentumAnalyzer

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Handles all market data retrieval for backtesting.

    Separates data access concerns from backtest logic,
    making each independently testable.
    """

    def __init__(self, cache: DataCache | None = None) -> None:
        """
        Initialize data fetcher.

        Args:
            cache: Optional DataCache instance (uses global if not provided)
        """
        self.cache = cache or get_cache()
        self.momentum_analyzer = MomentumAnalyzer(use_cache=True)

    def fetch_all_asset_data(
        self,
        years: int,
        extra_lookback_days: int = 400,
    ) -> pd.DataFrame:
        """
        Fetch price data for all VAA assets and core holdings.

        Args:
            years: Number of years of history to fetch
            extra_lookback_days: Additional days before start for momentum calc

        Returns:
            DataFrame of daily close prices (dates × tickers)
        """
        agg_tickers = list(ASSETS.AGGRESSIVE_TICKERS)
        prot_tickers = list(ASSETS.PROTECTIVE_TICKERS)
        core_tickers = ["SPY", "TLT", "GLD", "BIL"]
        all_tickers = list(set(agg_tickers + prot_tickers + core_tickers))

        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=years)
        fetch_start = start_date - pd.DateOffset(days=extra_lookback_days)

        logger.info("Fetching data for %s", all_tickers)
        price_data = self.cache.get_incremental_data(all_tickers, fetch_start, end_date)

        if price_data.empty:
            raise ValueError("No price data available for the requested period")

        return price_data

    def get_component_returns(
        self,
        years: int,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Compute monthly returns for VAA-selected asset and core holdings.

        Used by PortfolioOptimizer to determine optimal weight allocation.

        Args:
            years: Lookback period in years

        Returns:
            Tuple of (vaa_returns Series, core_returns DataFrame)
        """
        logger.info("Calculating component returns for %d-year period", years)

        agg_tickers = list(ASSETS.AGGRESSIVE_TICKERS)
        prot_tickers = list(ASSETS.PROTECTIVE_TICKERS)
        core_tickers = ["SPY", "TLT", "GLD", "BIL"]

        price_data = self.fetch_all_asset_data(years)

        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=years)

        monthly_prices = price_data.resample("ME").last()
        monthly_dates = monthly_prices.index[monthly_prices.index >= start_date]

        vaa_returns: list[float] = []
        core_returns_list: list[dict[str, float]] = []
        dates: list[pd.Timestamp] = []

        for i in range(len(monthly_dates) - 1):
            rebal_date = monthly_dates[i]
            next_date = monthly_dates[i + 1]

            hist_data = price_data.loc[:rebal_date]

            agg_mom_df = self.momentum_analyzer.calculate_momentum_series(hist_data, agg_tickers)
            if agg_mom_df.empty:
                continue

            prot_mom_df = self.momentum_analyzer.calculate_momentum_series(hist_data, prot_tickers)
            agg_current_scores = agg_mom_df.iloc[-1]
            is_defensive = bool((agg_current_scores < 0).any())

            if is_defensive:
                vaa_selected = (
                    prot_mom_df.iloc[-1].idxmax() if not prot_mom_df.empty else "SHY"
                )
            else:
                vaa_selected = agg_current_scores.idxmax()

            price_start = monthly_prices.loc[rebal_date]
            price_end = monthly_prices.loc[next_date]

            vaa_returns.append((price_end[vaa_selected] / price_start[vaa_selected]) - 1)

            core_ret = {
                asset: (price_end[asset] / price_start[asset]) - 1
                for asset in core_tickers
                if asset in price_start.index and asset in price_end.index
            }
            core_returns_list.append(core_ret)
            dates.append(next_date)

        vaa_series = pd.Series(vaa_returns, index=dates, name="VAA")
        core_df = pd.DataFrame(core_returns_list, index=dates)
        return vaa_series, core_df
