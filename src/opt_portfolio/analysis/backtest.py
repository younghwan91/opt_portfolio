"""
Backtesting Engine Module

Provides comprehensive backtesting capabilities for portfolio strategies.

퀀트 관점:
- 백테스트는 전략 검증의 필수 단계
- 과적합(overfitting) 주의: Out-of-sample 테스트 필요
- 거래비용, 슬리피지 등 현실적 가정 포함
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass, field

from ..config import BACKTEST, ASSETS, MOMENTUM, StrategyType
from ..core.cache import get_cache
from ..strategies.momentum import MomentumAnalyzer
from ..strategies.ou_process import OUForecaster
from .risk import RiskAnalyzer


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    initial_capital: float
    final_capital: float
    equity_curve: pd.Series
    returns: pd.Series
    transactions: List[Dict]
    
    # Calculated metrics
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    defensive_ratio: float = 0.0
    
    def calculate_metrics(self, years: float):
        """Calculate all performance metrics."""
        self.total_return = (self.final_capital / self.initial_capital) - 1
        self.cagr = (self.final_capital / self.initial_capital) ** (1/years) - 1
        
        if not self.returns.empty:
            risk_analyzer = RiskAnalyzer()
            self.volatility = risk_analyzer.calculate_volatility(self.returns)
            self.sharpe_ratio = risk_analyzer.calculate_sharpe_ratio(self.returns)
            self.max_drawdown, _, _ = risk_analyzer.calculate_max_drawdown(self.equity_curve)
            if self.max_drawdown > 0:
                self.calmar_ratio = self.cagr / self.max_drawdown
        
        # Win rate
        winning_months = (self.returns > 0).sum()
        total_months = len(self.returns)
        self.win_rate = winning_months / total_months if total_months > 0 else 0


class BacktestEngine:
    """
    Strategy backtesting engine.
    
    퀀트 조언:
    - 백테스트 기간은 최소 10년 (2개 이상의 경기 사이클)
    - Look-ahead bias 주의: 미래 데이터 사용 금지
    - Survivorship bias 주의: 상장폐지 종목 포함 필요
    - 거래비용 0.1%는 ETF 평균 수준
    """
    
    def __init__(
        self,
        initial_capital: float = BACKTEST.INITIAL_CAPITAL,
        transaction_cost: float = BACKTEST.TRANSACTION_COST
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as decimal
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.cache = get_cache()
        self.momentum_analyzer = MomentumAnalyzer(use_cache=True)
        self.forecaster = OUForecaster()
    
    def run_vaa_backtest(
        self,
        years: int = BACKTEST.DEFAULT_YEARS,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, BacktestResult]:
        """
        Run VAA backtest with multiple strategies.
        
        퀀트 조언:
        - Current: 기본 VAA (Keller 원본)
        - Forecast_1M: OU 1개월 예측 기반
        - Delta: 모멘텀 변화율 기반 (상승 추세 선호)
        
        Args:
            years: Backtest period in years
            strategies: List of strategy names to test
            
        Returns:
            Dictionary of strategy_name -> BacktestResult
        """
        if strategies is None:
            strategies = ['Current', 'Forecast_1M', 'Forecast_3M', 'Forecast_6M', 'Delta']
        
        print(f"🚀 Starting VAA Backtest ({years} years)")
        print("=" * 60)
        
        # Setup
        agg_tickers = list(ASSETS.AGGRESSIVE_TICKERS)
        prot_tickers = list(ASSETS.PROTECTIVE_TICKERS)
        all_tickers = list(set(agg_tickers + prot_tickers))
        
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=years)
        fetch_start = start_date - pd.DateOffset(days=400)
        
        # Fetch data
        print(f"📥 Fetching data for {all_tickers}...")
        price_data = self.cache.get_incremental_data(all_tickers, fetch_start, end_date)
        
        if price_data.empty:
            print("❌ No data found")
            return {}
        
        # Setup timeline
        monthly_prices = price_data.resample('ME').last()
        monthly_dates = monthly_prices.index[monthly_prices.index >= start_date]
        
        print(f"📅 Running backtest over {len(monthly_dates)} months...")
        
        # Initialize results
        results = {}
        capitals = {s: self.initial_capital for s in strategies}
        equity_curves = {s: [self.initial_capital] for s in strategies}
        monthly_returns = {s: [] for s in strategies}
        transactions_log = {s: [] for s in strategies}
        defensive_counts = {s: 0 for s in strategies}
        
        dates_recorded = []
        
        # Run backtest
        for i in range(len(monthly_dates) - 1):
            rebal_date = monthly_dates[i]
            next_date = monthly_dates[i + 1]
            dates_recorded.append(next_date)
            
            # Get historical data up to rebalance date
            hist_data = price_data.loc[:rebal_date]
            
            # Calculate momentum
            agg_mom_df = self.momentum_analyzer.calculate_momentum_series(hist_data, agg_tickers)
            if agg_mom_df.empty:
                continue
            
            prot_mom_df = self.momentum_analyzer.calculate_momentum_series(hist_data, prot_tickers)
            
            # Get scores for all strategies
            agg_scores = self._get_universe_scores(agg_mom_df)
            prot_scores = self._get_universe_scores(prot_mom_df) if not prot_mom_df.empty else {}
            
            # Select and calculate returns for each strategy
            for strategy in strategies:
                # Determine defensive mode
                safety_metric = 'Forecast_1M' if strategy == 'Delta' else strategy
                is_defensive = (agg_scores.get(safety_metric, agg_scores['Current']) < 0).any()
                
                if is_defensive:
                    defensive_counts[strategy] += 1
                    scores = prot_scores.get(strategy, prot_scores.get('Current', pd.Series()))
                else:
                    scores = agg_scores.get(strategy, agg_scores['Current'])
                
                if scores.empty:
                    continue
                
                selected = scores.idxmax()
                
                # Calculate return
                price_start = monthly_prices.loc[rebal_date, selected]
                price_end = monthly_prices.loc[next_date, selected]
                
                gross_return = (price_end / price_start) - 1
                net_return = gross_return - self.transaction_cost  # Apply transaction cost
                
                capitals[strategy] *= (1 + net_return)
                equity_curves[strategy].append(capitals[strategy])
                monthly_returns[strategy].append(net_return)
                
                transactions_log[strategy].append({
                    'date': next_date,
                    'asset': selected,
                    'mode': 'defensive' if is_defensive else 'aggressive',
                    'return': gross_return
                })
        
        # Create results
        for strategy in strategies:
            equity_series = pd.Series(
                equity_curves[strategy],
                index=[monthly_dates[0]] + dates_recorded[:len(equity_curves[strategy])-1]
            )
            returns_series = pd.Series(
                monthly_returns[strategy],
                index=dates_recorded[:len(monthly_returns[strategy])]
            )
            
            result = BacktestResult(
                strategy_name=strategy,
                initial_capital=self.initial_capital,
                final_capital=capitals[strategy],
                equity_curve=equity_series,
                returns=returns_series,
                transactions=transactions_log[strategy],
                defensive_ratio=defensive_counts[strategy] / len(monthly_dates) if len(monthly_dates) > 0 else 0
            )
            result.calculate_metrics(years)
            results[strategy] = result
        
        # Print summary
        self._print_backtest_summary(results, years)
        
        return results
    
    def _get_universe_scores(self, momentum_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate scores for all strategies."""
        if momentum_df.empty:
            return {}
        
        scores = {
            'Current': momentum_df.iloc[-1],
            'Forecast_1M': pd.Series(dtype=float),
            'Forecast_3M': pd.Series(dtype=float),
            'Forecast_6M': pd.Series(dtype=float),
            'Delta': pd.Series(dtype=float)
        }
        
        current_vals = momentum_df.iloc[-1]
        
        for ticker in momentum_df.columns:
            series = momentum_df[ticker].tail(MOMENTUM.CALIBRATION_WINDOW)
            
            f1 = self.forecaster.forecast(series, months=1)
            f3 = self.forecaster.forecast(series, months=3)
            f6 = self.forecaster.forecast(series, months=6)
            
            scores['Forecast_1M'][ticker] = f1
            scores['Forecast_3M'][ticker] = f3
            scores['Forecast_6M'][ticker] = f6
            scores['Delta'][ticker] = f1 - current_vals[ticker]
        
        return scores
    
    def _print_backtest_summary(self, results: Dict[str, BacktestResult], years: int):
        """Print backtest results summary."""
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("=" * 80)
        
        header = f"{'Strategy':<15} | {'Final Value':>12} | {'CAGR':>8} | {'Sharpe':>7} | {'MDD':>8} | {'Defensive':>10}"
        print(header)
        print("-" * 80)
        
        for name, result in results.items():
            print(f"{name:<15} | ${result.final_capital:>10,.0f} | {result.cagr:>7.1%} | "
                  f"{result.sharpe_ratio:>7.2f} | {result.max_drawdown:>7.1%} | {result.defensive_ratio:>9.1%}")
        
        print("=" * 80)
        
        # Find best strategy
        best_sharpe = max(results.values(), key=lambda x: x.sharpe_ratio)
        best_return = max(results.values(), key=lambda x: x.total_return)
        
        print(f"\n🏆 Best Sharpe Ratio: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})")
        print(f"💰 Best Total Return: {best_return.strategy_name} ({best_return.total_return:.1%})")
    
    def plot_results(
        self,
        results: Dict[str, BacktestResult],
        benchmark: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot backtest results.
        
        Args:
            results: Backtest results dictionary
            benchmark: Optional benchmark equity curve
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Equity curves
        ax1 = axes[0, 0]
        for name, result in results.items():
            ax1.plot(result.equity_curve.index, result.equity_curve.values, label=name, linewidth=2)
        
        if benchmark is not None:
            ax1.plot(benchmark.index, benchmark.values, label='Benchmark', 
                    linestyle='--', color='gray', linewidth=2)
        
        ax1.set_title('Equity Curves', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[0, 1]
        for name, result in results.items():
            running_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - running_max) / running_max * 100
            ax2.fill_between(drawdown.index, drawdown.values, alpha=0.3, label=name)
        
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Monthly returns distribution
        ax3 = axes[1, 0]
        for name, result in results.items():
            ax3.hist(result.returns * 100, bins=30, alpha=0.5, label=name)
        
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Monthly Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics comparison
        ax4 = axes[1, 1]
        metrics = ['CAGR', 'Sharpe', 'MDD', 'Win Rate']
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, (name, result) in enumerate(results.items()):
            values = [
                result.cagr * 100,
                result.sharpe_ratio,
                result.max_drawdown * 100,
                result.win_rate * 100
            ]
            ax4.bar(x + i * width, values, width, label=name)
        
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x + width * (len(results) - 1) / 2)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_benchmark(
        self,
        result: BacktestResult,
        benchmark_ticker: str = 'SPY'
    ) -> Dict:
        """
        Compare strategy with benchmark.
        
        퀀트 조언:
        - Alpha: 벤치마크 대비 초과 수익
        - Beta: 시장 민감도
        - Information Ratio: 추적오차 대비 초과 수익
        
        Args:
            result: Backtest result
            benchmark_ticker: Benchmark ticker
            
        Returns:
            Comparison metrics
        """
        # Get benchmark data
        start_date = result.equity_curve.index[0]
        end_date = result.equity_curve.index[-1]
        
        benchmark_data = self.cache.get_incremental_data(
            [benchmark_ticker], start_date, end_date
        )
        
        if benchmark_data.empty:
            return {}
        
        benchmark_monthly = benchmark_data.resample('ME').last()
        benchmark_returns = benchmark_monthly.pct_change().dropna()[benchmark_ticker]
        
        risk_analyzer = RiskAnalyzer()
        
        # Align returns
        common_dates = result.returns.index.intersection(benchmark_returns.index)
        strategy_returns = result.returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calculate comparison metrics
        beta = risk_analyzer.calculate_beta(strategy_returns, benchmark_returns)
        
        strategy_annual = strategy_returns.mean() * 12
        benchmark_annual = benchmark_returns.mean() * 12
        alpha = strategy_annual - (beta * benchmark_annual)
        
        tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(12)
        information_ratio = (strategy_annual - benchmark_annual) / tracking_error if tracking_error > 0 else 0
        
        return {
            'strategy_cagr': result.cagr,
            'benchmark_cagr': (1 + benchmark_annual) - 1,
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'correlation': strategy_returns.corr(benchmark_returns)
        }
