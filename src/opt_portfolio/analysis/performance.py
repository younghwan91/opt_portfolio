"""
Performance Analysis Module

Provides performance attribution and analysis tools.

퀀트 관점:
- 성과 분석은 전략 개선의 핵심
- 수익 원천(return attribution) 파악 필요
- 시장 환경별 성과 분석으로 전략 특성 이해
"""

import numpy as np
import pandas as pd

from .metrics import calculate_cagr
from .risk import RiskAnalyzer


class PerformanceAnalyzer:
    """
    Performance analysis and attribution.

    퀀트 조언:
    - CAGR (복리 연환산 수익률)이 투자 성과의 핵심 지표
    - 수익률 분포 분석으로 전략 특성 파악
    - 시장 환경별 분석으로 강점/약점 파악
    """

    def __init__(self) -> None:
        """Initialize performance analyzer."""
        self.risk_analyzer = RiskAnalyzer()

    def calculate_cagr(self, initial_value: float, final_value: float, years: float) -> float:
        """
        Calculate Compound Annual Growth Rate.

        퀀트 조언:
        - CAGR = (최종가치/초기가치)^(1/년수) - 1
        - 단순 평균 수익률보다 정확한 성과 측정
        - 복리 효과 반영

        Args:
            initial_value: Starting value
            final_value: Ending value
            years: Investment period in years

        Returns:
            CAGR as decimal
        """
        return calculate_cagr(initial_value, final_value, years)

    def calculate_rolling_returns(self, returns: pd.Series, window: int = 12) -> pd.Series:
        """
        Calculate rolling returns.

        Args:
            returns: Returns series
            window: Rolling window in periods

        Returns:
            Rolling returns series
        """
        return (1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=True)

    def calculate_monthly_statistics(self, returns: pd.Series) -> dict:
        """
        Calculate monthly return statistics.

        퀀트 조언:
        - 월별 통계로 수익률 분포 특성 파악
        - 왜도(skewness) > 0이면 우측 꼬리 (좋음)
        - 첨도(kurtosis) > 3이면 꼬리 위험 존재

        Args:
            returns: Monthly returns series

        Returns:
            Dictionary of statistics
        """
        return {
            "mean": returns.mean(),
            "median": returns.median(),
            "std": returns.std(),
            "min": returns.min(),
            "max": returns.max(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "positive_months": (returns > 0).sum(),
            "negative_months": (returns < 0).sum(),
            "win_rate": (returns > 0).mean(),
            "best_month": returns.max(),
            "worst_month": returns.min(),
            "avg_gain": returns[returns > 0].mean() if (returns > 0).any() else 0,
            "avg_loss": returns[returns < 0].mean() if (returns < 0).any() else 0,
        }

    def analyze_by_year(self, returns: pd.Series) -> pd.DataFrame:
        """
        Analyze returns by calendar year.

        퀀트 조언:
        - 연도별 성과로 일관성 파악
        - 특정 연도 의존도 체크
        - 시장 환경(강세/약세)별 성과 비교

        Args:
            returns: Returns series with datetime index

        Returns:
            DataFrame with annual statistics
        """
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)

        annual_data = []

        for year in returns.index.year.unique():
            year_returns = returns[returns.index.year == year]

            annual_return = (1 + year_returns).prod() - 1
            vol = year_returns.std() * np.sqrt(12)

            annual_data.append(
                {
                    "Year": year,
                    "Return": annual_return,
                    "Volatility": vol,
                    "Sharpe": annual_return / vol if vol > 0 else 0,
                    "Best Month": year_returns.max(),
                    "Worst Month": year_returns.min(),
                    "Positive Months": (year_returns > 0).sum(),
                }
            )

        return pd.DataFrame(annual_data)

    def analyze_by_market_regime(
        self, strategy_returns: pd.Series, benchmark_returns: pd.Series
    ) -> dict:
        """
        Analyze performance by market regime.

        퀀트 조언:
        - 상승장(Up Market): 벤치마크 > 0
        - 하락장(Down Market): 벤치마크 < 0
        - 좋은 전략은 하락장에서 방어력 발휘
        - Capture Ratio로 측정: Up Capture > 100%, Down Capture < 100%

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Dictionary with regime analysis
        """
        # Align returns
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strat = strategy_returns.loc[common_idx]
        bench = benchmark_returns.loc[common_idx]

        # Define regimes
        up_market = bench > 0
        down_market = bench < 0

        # Calculate regime statistics
        up_capture = strat[up_market].mean() / bench[up_market].mean() if up_market.any() else 0
        down_capture = (
            strat[down_market].mean() / bench[down_market].mean() if down_market.any() else 0
        )

        return {
            "up_market": {
                "count": up_market.sum(),
                "strategy_avg": strat[up_market].mean(),
                "benchmark_avg": bench[up_market].mean(),
                "capture_ratio": up_capture * 100,
            },
            "down_market": {
                "count": down_market.sum(),
                "strategy_avg": strat[down_market].mean(),
                "benchmark_avg": bench[down_market].mean(),
                "capture_ratio": down_capture * 100,
            },
            "overall": {
                "correlation": strat.corr(bench),
                "up_capture": up_capture * 100,
                "down_capture": down_capture * 100,
                "capture_ratio": (up_capture - down_capture) * 100,
            },
        }

    def calculate_drawdown_analysis(self, equity_curve: pd.Series, top_n: int = 5) -> pd.DataFrame:
        """
        Analyze top drawdown periods.

        퀀트 조언:
        - 낙폭 분석으로 최악의 시나리오 이해
        - 회복 기간은 투자자 인내심 테스트
        - 낙폭 패턴으로 전략 특성 파악

        Args:
            equity_curve: Equity curve series
            top_n: Number of top drawdowns to analyze

        Returns:
            DataFrame with drawdown details
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

        start_dates = equity_curve.index[drawdown_starts]
        end_dates = equity_curve.index[drawdown_ends]

        # Match starts with ends
        drawdown_periods = []

        for start in start_dates:
            # Find corresponding end
            possible_ends = end_dates[end_dates > start]
            if len(possible_ends) > 0:
                end = possible_ends[0]
            else:
                end = equity_curve.index[-1]

            # Find trough
            period_dd = drawdown[start:end]
            trough_date = period_dd.idxmin()
            max_dd = period_dd.min()

            drawdown_periods.append(
                {
                    "Start": start,
                    "Trough": trough_date,
                    "End": end,
                    "Max Drawdown": max_dd,
                    "Days to Trough": (trough_date - start).days,
                    "Days to Recovery": (end - trough_date).days,
                    "Total Duration": (end - start).days,
                }
            )

        df = pd.DataFrame(drawdown_periods)
        df = df.sort_values("Max Drawdown").head(top_n)
        df["Max Drawdown"] = df["Max Drawdown"] * 100  # Convert to percentage

        return df.reset_index(drop=True)

    def generate_performance_report(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> str:
        """
        Generate comprehensive performance report.

        Args:
            equity_curve: Equity curve
            returns: Returns series
            benchmark_returns: Optional benchmark returns

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("📊 PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)

        # Basic statistics
        stats = self.calculate_monthly_statistics(returns)

        # Calculate period
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        cagr = self.calculate_cagr(equity_curve.iloc[0], equity_curve.iloc[-1], years)

        report.append(
            f"\n📅 Period: {equity_curve.index[0].strftime('%Y-%m-%d')} to "
            f"{equity_curve.index[-1].strftime('%Y-%m-%d')}"
        )
        report.append(f"   Duration: {years:.1f} years")

        report.append("\n💰 RETURNS:")
        report.append(f"   Total Return: {total_return:.1%}")
        report.append(f"   CAGR: {cagr:.1%}")
        report.append(f"   Best Month: {stats['best_month']:.1%}")
        report.append(f"   Worst Month: {stats['worst_month']:.1%}")

        report.append("\n📊 RISK METRICS:")
        report.append(f"   Volatility: {stats['std'] * np.sqrt(12):.1%}")
        report.append(f"   Sharpe Ratio: {self.risk_analyzer.calculate_sharpe_ratio(returns):.2f}")

        max_dd, _, _ = self.risk_analyzer.calculate_max_drawdown(equity_curve)
        report.append(f"   Max Drawdown: {max_dd:.1%}")

        report.append("\n📈 CONSISTENCY:")
        report.append(f"   Win Rate: {stats['win_rate']:.1%}")
        report.append(f"   Positive Months: {stats['positive_months']}")
        report.append(f"   Negative Months: {stats['negative_months']}")
        report.append(f"   Avg Gain: {stats['avg_gain']:.1%}")
        report.append(f"   Avg Loss: {stats['avg_loss']:.1%}")

        if benchmark_returns is not None:
            report.append("\n📊 VS BENCHMARK:")
            regime = self.analyze_by_market_regime(returns, benchmark_returns)
            report.append(f"   Correlation: {regime['overall']['correlation']:.2f}")
            report.append(f"   Up Capture: {regime['overall']['up_capture']:.1f}%")
            report.append(f"   Down Capture: {regime['overall']['down_capture']:.1f}%")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
