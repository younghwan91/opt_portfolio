#!/usr/bin/env python3
"""
Optimal Portfolio Management System - Main Entry Point

This is the main entry point for the portfolio management system.
Run this file to access the full menu of options.

Usage:
    python main.py          # Interactive menu
    python main.py --web    # Launch web UI directly
    python main.py --cli    # Launch CLI directly
"""

import subprocess
import sys
import argparse
from pathlib import Path


def launch_web_ui():
    """Launch the Streamlit web interface."""
    print("\n🌐 Launching Web UI...")
    app_path = Path(__file__).parent / "src" / "opt_portfolio" / "ui" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def launch_cli():
    """Launch the command line interface."""
    print("\n💻 Launching CLI...")
    from src.opt_portfolio.ui.cli import main
    main()


def run_vaa_analysis():
    """Run VAA analysis directly."""
    from src.opt_portfolio.strategies.vaa import VAAStrategy
    from datetime import date
    
    print("\n📊 Running VAA Analysis...")
    vaa = VAAStrategy(use_forecasting=True)
    result = vaa.select(date.today())
    
    print(f"\n🎯 Selected ETF: {result.selected_etf}")
    return result.selected_etf


def run_backtest():
    """Run strategy backtest."""
    from src.opt_portfolio.analysis.backtest import BacktestEngine
    
    print("\n📈 Running Backtest...")
    engine = BacktestEngine()
    results = engine.run_vaa_backtest(years=15)
    engine.plot_results(results)


def plot_momentum():
    """Plot momentum history."""
    from src.opt_portfolio.strategies.momentum import MomentumAnalyzer
    from src.opt_portfolio.utils.visualization import create_momentum_chart
    from src.opt_portfolio.config import ASSETS
    from datetime import date, timedelta
    import matplotlib.pyplot as plt
    
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
    start_date = end_date - timedelta(days=years*365)
    
    momentum_df = analyzer.calculate_historical_momentum(tickers, start_date, end_date)
    
    if not momentum_df.empty:
        fig = create_momentum_chart(momentum_df, f"{title} Assets Momentum")
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
    parser.add_argument("--web", action="store_true", help="Launch web UI")
    parser.add_argument("--cli", action="store_true", help="Launch CLI")
    args = parser.parse_args()
    
    if args.web:
        launch_web_ui()
        return
    
    if args.cli:
        launch_cli()
        return
    
    print("\n" + "=" * 50)
    print("🚀 OPTIMAL PORTFOLIO MANAGEMENT SYSTEM")
    print("=" * 50)
    print("VAA Strategy with Advanced OU Forecasting")
    print()
    
    while True:
        print("\nChoose an option:")
        print("1. 🌐 Launch Web UI (Recommended)")
        print("2. 💻 Launch CLI")
        print("3. 📊 Quick VAA Analysis")
        print("4. 📈 Run Backtest")
        print("5. 📉 Plot Momentum History")
        print("6. 💾 Cache Management")
        print("7. ❌ Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            launch_web_ui()
        elif choice == "2":
            launch_cli()
        elif choice == "3":
            run_vaa_analysis()
        elif choice == "4":
            run_backtest()
        elif choice == "5":
            plot_momentum()
        elif choice == "6":
            cache_management()
        elif choice == "7":
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
