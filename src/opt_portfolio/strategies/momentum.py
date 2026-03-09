"""
Momentum Analysis Module

This module provides momentum calculation and analysis functionality
for the VAA strategy and other momentum-based approaches.

퀀트 관점:
- 모멘텀은 "과거 승자가 미래에도 승리"하는 시장 이상현상
- 학술적으로 검증된 가장 강력한 팩터 중 하나
- 다만 모멘텀 붕괴(momentum crash) 위험 존재 (2009년 3월 등)
"""

import logging
from datetime import date, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta

from ..config import MOMENTUM
from ..core.cache import get_cache

logger = logging.getLogger(__name__)


class MomentumAnalyzer:
    """
    Momentum analysis and calculation engine.

    퀀트 조언:
    - VAA의 모멘텀 공식: 12×(1개월) + 4×(3개월) + 2×(6개월) + 1×(12개월)
    - 단기 수익률에 높은 가중치 = 빠른 시장 반응
    - 절대 모멘텀(0 기준)과 상대 모멘텀(자산 간 비교) 모두 사용
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize momentum analyzer.

        Args:
            use_cache: Whether to use data caching
        """
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None

    def calculate_returns(
        self, prices: pd.DataFrame, periods: list[int] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Calculate returns for multiple periods.

        Args:
            prices: DataFrame with dates as index and tickers as columns
            periods: List of periods in trading days (default: 1m, 3m, 6m, 12m)

        Returns:
            Dictionary of {period_name: returns_dataframe}
        """
        if periods is None:
            periods = [
                MOMENTUM.PERIOD_1M_DAYS,
                MOMENTUM.PERIOD_3M_DAYS,
                MOMENTUM.PERIOD_6M_DAYS,
                MOMENTUM.PERIOD_12M_DAYS,
            ]

        returns = {}
        period_names = ["1-Month", "3-Month", "6-Month", "12-Month"]

        for days, name in zip(periods, period_names):
            returns[name] = prices.pct_change(days) * 100

        return returns

    def calculate_momentum_score(
        self, returns: dict[str, pd.DataFrame], weights: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Calculate weighted momentum scores.

        퀀트 조언:
        - 가중치 (12, 4, 2, 1)은 VAA 논문 기반
        - 합이 19이므로 최대 가능 점수 = 19 × 100% = 1900
        - 실제로는 -200 ~ +500 범위가 일반적

        Args:
            returns: Dictionary of period returns
            weights: Momentum weights (default: [12, 4, 2, 1])

        Returns:
            DataFrame with momentum scores
        """
        if weights is None:
            weights = [
                MOMENTUM.WEIGHT_1M,
                MOMENTUM.WEIGHT_3M,
                MOMENTUM.WEIGHT_6M,
                MOMENTUM.WEIGHT_12M,
            ]

        period_names = ["1-Month", "3-Month", "6-Month", "12-Month"]

        momentum = pd.DataFrame(index=returns[period_names[0]].index)

        for ticker in returns[period_names[0]].columns:
            momentum[ticker] = sum(
                returns[name][ticker] * weight for name, weight in zip(period_names, weights)
            )

        return momentum.dropna()

    def get_performance(self, tickers: list[str], end_date: date) -> pd.DataFrame:
        """
        Calculate periodic performance for tickers as of a given date.

        Args:
            tickers: List of ticker symbols
            end_date: Reference date for calculation

        Returns:
            DataFrame with periodic returns for each ticker
        """
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - relativedelta(months=13)  # 12 months + buffer

        # Get price data
        if self.cache:
            data = self.cache.get_incremental_data(tickers, start_dt, end_dt)
        else:
            import yfinance as yf

            data = yf.download(
                tickers, start=start_dt, end=end_dt + timedelta(days=1), auto_adjust=True
            )["Close"]

        if data.empty:
            return pd.DataFrame()

        # Calculate performance for each period
        performance_data = {}
        periods_months = [1, 3, 6, 12]

        for ticker in tickers:
            if ticker not in data.columns:
                continue

            ticker_prices = data[ticker].dropna()
            if ticker_prices.empty:
                continue

            actual_end = ticker_prices.index.max()
            end_price = ticker_prices.loc[actual_end]

            returns = {}
            for months in periods_months:
                start_date_period = actual_end - relativedelta(months=months)
                try:
                    start_price = ticker_prices.asof(start_date_period)
                    if pd.notna(start_price) and start_price > 0:
                        returns[f"{months}-Month"] = round(((end_price / start_price) - 1) * 100, 2)
                    else:
                        returns[f"{months}-Month"] = None
                except Exception:
                    returns[f"{months}-Month"] = None

            performance_data[ticker] = returns

        df = pd.DataFrame(performance_data).T
        if not df.empty:
            df = df[[f"{p}-Month" for p in periods_months]]

        return df

    def calculate_and_rank(self, tickers: list[str], end_date: date) -> tuple[pd.DataFrame, str]:
        """
        Calculate momentum and return ranked results with top pick.

        Args:
            tickers: List of ticker symbols
            end_date: Reference date

        Returns:
            Tuple of (ranked_dataframe, top_ticker)
        """
        performance = self.get_performance(tickers, end_date)

        if performance.empty:
            return pd.DataFrame(), None

        # Ensure numeric columns
        for col in ["1-Month", "3-Month", "6-Month", "12-Month"]:
            performance[col] = pd.to_numeric(performance[col], errors="coerce")

        performance.dropna(subset=["1-Month", "3-Month", "6-Month", "12-Month"], inplace=True)

        if performance.empty:
            return pd.DataFrame(), None

        # Calculate momentum score
        performance["Momentum Score"] = (
            performance["1-Month"] * MOMENTUM.WEIGHT_1M
            + performance["3-Month"] * MOMENTUM.WEIGHT_3M
            + performance["6-Month"] * MOMENTUM.WEIGHT_6M
            + performance["12-Month"] * MOMENTUM.WEIGHT_12M
        )

        # Rank by momentum
        ranked = performance.sort_values("Momentum Score", ascending=False)
        top_ticker = ranked.index[0] if not ranked.empty else None

        return ranked, top_ticker

    def calculate_historical_momentum(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Calculate historical momentum scores over a date range.

        퀀트 조언:
        - 히스토리컬 모멘텀은 백테스트와 시계열 분석에 필수
        - 롤링 계산으로 매일의 모멘텀 점수 확보

        Args:
            tickers: List of ticker symbols
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            DataFrame with dates as index and momentum scores as columns
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Need extra data before start for momentum calculation
        fetch_start = start_dt - timedelta(days=400)

        logger.info("Calculating historical momentum from %s to %s...", start_date, end_date)

        try:
            if self.cache:
                data = self.cache.get_incremental_data(tickers, fetch_start, end_dt)
            else:
                import yfinance as yf

                data = yf.download(
                    tickers, start=fetch_start, end=end_dt + timedelta(days=1), auto_adjust=True
                )["Close"]

            if data.empty:
                return pd.DataFrame()

            # Calculate rolling returns
            r1 = data.pct_change(MOMENTUM.PERIOD_1M_DAYS) * 100
            r3 = data.pct_change(MOMENTUM.PERIOD_3M_DAYS) * 100
            r6 = data.pct_change(MOMENTUM.PERIOD_6M_DAYS) * 100
            r12 = data.pct_change(MOMENTUM.PERIOD_12M_DAYS) * 100

            # Calculate momentum scores
            momentum_scores = (
                r1 * MOMENTUM.WEIGHT_1M
                + r3 * MOMENTUM.WEIGHT_3M
                + r6 * MOMENTUM.WEIGHT_6M
                + r12 * MOMENTUM.WEIGHT_12M
            )

            # Filter for requested date range
            momentum_scores = momentum_scores.loc[start_dt:end_dt]

            return momentum_scores.dropna()

        except Exception as e:
            logger.error("Error calculating historical momentum: %s", e)
            return pd.DataFrame()

    def calculate_momentum_series(
        self, hist_data: pd.DataFrame, tickers: list[str], lookback_days: int = 60
    ) -> pd.DataFrame:
        """
        Calculate momentum score series for backtesting.

        Args:
            hist_data: Historical price data
            tickers: Tickers to analyze
            lookback_days: How many days of momentum to return

        Returns:
            DataFrame with momentum series
        """
        # Need at least 252 days for 12-month return + buffer
        recent_data = hist_data.tail(MOMENTUM.PERIOD_12M_DAYS + lookback_days)

        if len(recent_data) < MOMENTUM.PERIOD_12M_DAYS:
            return pd.DataFrame()

        momentum_data = {}

        for ticker in tickers:
            if ticker not in recent_data.columns:
                continue

            prices = recent_data[ticker]

            # Calculate rolling returns
            r1 = prices.pct_change(MOMENTUM.PERIOD_1M_DAYS) * 100
            r3 = prices.pct_change(MOMENTUM.PERIOD_3M_DAYS) * 100
            r6 = prices.pct_change(MOMENTUM.PERIOD_6M_DAYS) * 100
            r12 = prices.pct_change(MOMENTUM.PERIOD_12M_DAYS) * 100

            # Weighted momentum
            mom = (
                r1 * MOMENTUM.WEIGHT_1M
                + r3 * MOMENTUM.WEIGHT_3M
                + r6 * MOMENTUM.WEIGHT_6M
                + r12 * MOMENTUM.WEIGHT_12M
            )

            momentum_data[ticker] = mom.dropna().tail(lookback_days)

        return pd.DataFrame(momentum_data)

    def is_negative_momentum(self, ranked_df: pd.DataFrame) -> bool:
        """
        Check if any asset has negative momentum.

        퀀트 조언:
        - VAA의 핵심 방어 로직
        - 하나라도 음수면 "위험 신호"로 간주
        - 보수적 접근이지만 큰 하락장에서 효과적

        Args:
            ranked_df: Ranked momentum DataFrame

        Returns:
            True if any momentum score is negative
        """
        if "Momentum Score" not in ranked_df.columns:
            return False

        return (ranked_df["Momentum Score"] < 0).any()
