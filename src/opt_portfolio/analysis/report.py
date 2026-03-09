"""
Report Generation Module for Backtest Engine

Encapsulates all formatted output and result presentation logic,
keeping the backtest computation classes free of display concerns.
"""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult


class BacktestReporter:
    """
    Generates human-readable reports from backtest results.

    Separates presentation from computation — the BacktestEngine
    delegates all printing/formatting to this class.
    """

    def print_dynamic_summary(self, result: BacktestResult, years: int) -> None:
        """
        Print a formatted summary of a dynamic VAA backtest result.

        Args:
            result: Completed backtest result
            years: Backtest period in years
        """
        print("\n" + "=" * 70)
        print("📊 DYNAMIC VAA BACKTEST RESULT")
        print("=" * 70)
        print(f"{'Metric':<25} | {'Value':<20}")
        print("-" * 70)
        print(f"{'Strategy':<25} | {result.strategy_name:<20}")
        print(f"{'Period':<25} | {years} years")
        print(f"{'Initial Capital':<25} | ${result.initial_capital:,.0f}")
        print(f"{'Final Capital':<25} | ${result.final_capital:,.0f}")
        print(f"{'Total Return':<25} | {result.total_return:.1%}")
        print(f"{'CAGR':<25} | {result.cagr:.1%}")
        print(f"{'Sharpe Ratio':<25} | {result.sharpe_ratio:.3f}")
        print(f"{'Max Drawdown':<25} | {result.max_drawdown:.1%}")
        print(f"{'Win Rate':<25} | {result.win_rate:.1%}")
        print(f"{'Defensive Months':<25} | {result.defensive_ratio:.1%}")

        if result.vaa_selections:
            print("\n📈 VAA Selection Distribution:")
            selection_counts = pd.Series(result.vaa_selections).value_counts()
            for ticker, count in selection_counts.items():
                pct = count / len(result.vaa_selections) * 100
                print(f"   {ticker}: {count} months ({pct:.1f}%)")

        print("=" * 70)

    def print_comparison_summary(self, results: dict[str, BacktestResult], years: int) -> None:
        """
        Print a side-by-side comparison of multiple backtest results.

        Args:
            results: Dictionary of strategy name → BacktestResult
            years: Backtest period in years
        """
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("=" * 80)

        header = (
            f"{'Strategy':<15} | {'Final Value':>12} | {'CAGR':>8} | "
            f"{'Sharpe':>7} | {'MDD':>8} | {'Defensive':>10}"
        )
        print(header)
        print("-" * 80)

        for name, result in results.items():
            print(
                f"{name:<15} | ${result.final_capital:>10,.0f} | {result.cagr:>7.1%} | "
                f"{result.sharpe_ratio:>7.2f} | {result.max_drawdown:>7.1%} | "
                f"{result.defensive_ratio:>9.1%}"
            )

        print("=" * 80)

        best_sharpe = max(results.values(), key=lambda x: x.sharpe_ratio)
        best_return = max(results.values(), key=lambda x: x.total_return)

        print(
            f"\n🏆 Best Sharpe Ratio: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})"
        )
        print(f"💰 Best Total Return: {best_return.strategy_name} ({best_return.total_return:.1%})")
