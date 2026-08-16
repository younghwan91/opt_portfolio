#!/usr/bin/env python3
"""
Optimal Portfolio Management System - Main Entry Point

VAA (Vigilant Asset Allocation) 전략 기반 포트폴리오 관리 시스템
동적 VAA 선택 + Sharpe Ratio 기반 비중 최적화 지원

Usage:
    python run.py              # Interactive menu
    python run.py --cli        # Launch CLI
    python run.py --backtest   # Run dynamic backtest
    python run.py --optimize   # Run optimization
"""

import argparse


def launch_cli():
    """Launch the command line interface."""
    print("\n💻 Launching CLI...")
    from src.opt_portfolio.ui.cli import main

    main()


def run_vaa_analysis():
    """Run VAA analysis directly."""
    from datetime import date

    from src.opt_portfolio.strategies.vaa import VAAStrategy

    print("\n📊 Running VAA Analysis...")
    vaa = VAAStrategy(use_forecasting=True)
    result = vaa.select(date.today())

    print(f"\n🎯 Selected ETF: {result.selected_etf}")
    return result.selected_etf


def run_dynamic_backtest():
    """Run dynamic VAA backtest with default weights."""
    from src.opt_portfolio.analysis.backtest import BacktestEngine

    print("\n📈 Running Dynamic VAA Backtest...")
    engine = BacktestEngine()

    # Get user input for years
    try:
        years_input = input("Enter backtest period in years (default 15): ").strip()
        years = int(years_input) if years_input else 15
    except ValueError:
        years = 15

    result = engine.run_dynamic_vaa_backtest(years=years)

    # Optionally plot
    show_plot = input("\nShow equity curve plot? (y/n, default n): ").strip().lower()
    if show_plot == "y":
        engine.plot_results({"Dynamic VAA": result})

    return result


def run_optimized_backtest():
    """Run backtest with Sharpe Ratio optimization."""
    from src.opt_portfolio.analysis.backtest import BacktestEngine

    print("\n🔬 Running Optimized Backtest...")
    print("This will find the optimal portfolio weights based on Sharpe Ratio.")

    engine = BacktestEngine()

    # Get user input for years
    try:
        years_input = input("Enter backtest period in years (default 15): ").strip()
        years = int(years_input) if years_input else 15
    except ValueError:
        years = 15

    result, opt_result = engine.run_optimized_backtest(years=years)

    print("\n" + opt_result.get_summary())

    # Optionally plot
    show_plot = input("\nShow equity curve plot? (y/n, default n): ").strip().lower()
    if show_plot == "y":
        engine.plot_results({"Optimized VAA": result})

    return result, opt_result


def run_strategy_comparison():
    """Run comparison of different VAA strategies."""
    from src.opt_portfolio.analysis.backtest import BacktestEngine

    print("\n📊 Running VAA Strategy Comparison...")
    engine = BacktestEngine()

    try:
        years_input = input("Enter backtest period in years (default 15): ").strip()
        years = int(years_input) if years_input else 15
    except ValueError:
        years = 15

    results = engine.run_vaa_backtest(years=years)

    # Plot comparison
    show_plot = input("\nShow comparison plot? (y/n, default n): ").strip().lower()
    if show_plot == "y":
        engine.plot_results(results)

    return results


def plot_momentum():
    """Plot momentum history."""
    from datetime import date, timedelta

    import matplotlib.pyplot as plt

    from src.opt_portfolio.config import ASSETS
    from src.opt_portfolio.strategies.momentum import MomentumAnalyzer

    print("\n📉 VAA MOMENTUM HISTORY")
    print("-" * 40)

    print("1. Aggressive Assets (SPY, EFA, EEM, AGG)")
    print("2. Protective Assets (LQD, IEF, SHY)")

    choice = input("Choose (1-2): ").strip()

    tickers = list(ASSETS.AGGRESSIVE_TICKERS) if choice == "1" else list(ASSETS.PROTECTIVE_TICKERS)
    title = "Aggressive" if choice == "1" else "Protective"

    years = int(input("History in years (default 2): ") or "2")

    analyzer = MomentumAnalyzer()
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)

    momentum_df = analyzer.calculate_historical_momentum(tickers, start_date, end_date)

    if not momentum_df.empty:
        plt.figure(figsize=(12, 6))
        for col in momentum_df.columns:
            plt.plot(momentum_df.index, momentum_df[col], label=col, linewidth=2)

        plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        plt.title(f"{title} Assets Momentum History")
        plt.xlabel("Date")
        plt.ylabel("Momentum Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No data available.")


def cache_management():
    """Cache management menu."""
    from src.opt_portfolio.core.cache import get_cache

    print("\n💾 CACHE MANAGEMENT")
    print("-" * 40)
    print("1. View statistics")
    print("2. Clear all")
    print("3. Optimize")

    choice = input("Choose (1-3): ").strip()
    cache = get_cache()

    if choice == "1":
        stats = cache.get_cache_stats()
        if stats.empty:
            print("No cached data.")
        else:
            print(stats.to_string())
    elif choice == "2":
        if input("Clear ALL? (yes/no): ").lower() == "yes":
            cache.clear_cache()
    elif choice == "3":
        cache.optimize()


def main():
    """Main entry point with menu."""
    parser = argparse.ArgumentParser(description="Optimal Portfolio Management System")
    parser.add_argument("--cli", action="store_true", help="Launch CLI")
    parser.add_argument("--backtest", action="store_true", help="Run dynamic backtest")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    args = parser.parse_args()

    if args.cli:
        launch_cli()
        return

    if args.backtest:
        run_dynamic_backtest()
        return

    if args.optimize:
        run_optimized_backtest()
        return

    print("\n" + "=" * 60)
    print("🚀 OPTIMAL PORTFOLIO MANAGEMENT SYSTEM")
    print("   VAA Strategy with Dynamic Selection & Weight Optimization")
    print("=" * 60)

    while True:
        print("\nChoose an option:")
        print("1. 💻 Launch CLI")
        print("2. 📊 Quick VAA Analysis")
        print("3. 📈 Run Dynamic VAA Backtest")
        print("4. 🔬 Run Optimized Backtest (Sharpe Ratio)")
        print("5. 📉 Compare VAA Strategies")
        print("6. 📊 Plot Momentum History")
        print("7. 💾 Cache Management")
        print("8. ❌ Exit")

        choice = input("\nEnter choice (1-8): ").strip()

        if choice == "1":
            launch_cli()
        elif choice == "2":
            run_vaa_analysis()
        elif choice == "3":
            run_dynamic_backtest()
        elif choice == "4":
            run_optimized_backtest()
        elif choice == "5":
            run_strategy_comparison()
        elif choice == "6":
            plot_momentum()
        elif choice == "7":
            cache_management()
        elif choice == "8":
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
