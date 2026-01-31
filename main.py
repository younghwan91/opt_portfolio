import subprocess
import sys
from data_cache import get_cache
from datetime import date, timedelta
import vaa_agg

# Run when __name__ == "__main__"
if __name__ == "__main__":
    print("🚀 PORTFOLIO MANAGEMENT SYSTEM")
    print("=" * 40)
    print("Choose an option:")
    print("1. Run integrated portfolio management (CLI)")
    print("2. Launch web UI (Streamlit)")
    print("3. Run VAA analysis only")
    print("4. Run portfolio calculator only")
    print("5. Plot VAA Momentum History")
    print("6. Cache management")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        print("\nLaunching integrated portfolio management...")
        subprocess.run([sys.executable, "integrated_portfolio.py"])
    elif choice == '2':
        print("\nLaunching web UI...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "portfolio_ui.py"])
    elif choice == '3':
        print("\nRunning VAA Aggregation...")
        subprocess.run([sys.executable, "vaa_agg.py"])
    elif choice == '4':
        print("\nRunning Portfolio Ratio Calculator...")
        subprocess.run([sys.executable, "port_ratio_calculator.py"])
    elif choice == '5':
        print("\n📈 VAA MOMENTUM HISTORY PLOT")
        print("=" * 40)
        
        # Default tickers
        aggressive_tickers = ['SPY', 'EFA', 'EEM', 'AGG']
        protective_tickers = ['LQD', 'IEF', 'SHY']
        
        print("1. Plot Aggressive Assets (SPY, EFA, EEM, AGG)")
        print("2. Plot Protective Assets (LQD, IEF, SHY)")
        print("3. Custom Tickers")
        
        plot_choice = input("Choose (1-3): ").strip()
        
        tickers = []
        if plot_choice == '1':
            tickers = aggressive_tickers
            title = "Aggressive Assets Momentum"
        elif plot_choice == '2':
            tickers = protective_tickers
            title = "Protective Assets Momentum"
        elif plot_choice == '3':
            t_str = input("Enter tickers (comma-separated): ").strip()
            tickers = [t.strip().upper() for t in t_str.split(',')]
            title = "Custom Assets Momentum"
        
        if tickers:
            years = input("Enter history duration in years (default 2): ").strip()
            years = int(years) if years.isdigit() else 2
            
            end_date = date.today()
            start_date = end_date - timedelta(days=years*365)
            
            print(f"\nCalculating momentum from {start_date} to {end_date}...")
            momentum_df = vaa_agg.calculate_historical_momentum(tickers, start_date, end_date)
            
            if not momentum_df.empty:
                # Ask for forecast
                forecast_months = input("Forecast future momentum? Enter months (0 to skip, default 1): ").strip()
                forecast_months = int(forecast_months) if forecast_months.isdigit() else 1
                
                forecast_df = None
                win_probs = None
                
                if forecast_months > 0:
                    win_probs, forecast_df = vaa_agg.simulate_momentum_ou(momentum_df, months=forecast_months)
                    if not win_probs.empty:
                        print("\n🏆 Probability of being the Best Asset (Next Month):")
                        for ticker, prob in win_probs.items():
                            print(f"  {ticker}: {prob*100:.1f}%")
                
                vaa_agg.plot_momentum_history(momentum_df, title, forecast_df, win_probs)
            else:
                print("No data available to plot.")
                
    elif choice == '6':
        print("\n📦 CACHE MANAGEMENT")
        print("=" * 40)
        print("1. View cache statistics")
        print("2. Clear all cache")
        print("3. Clear cache older than 30 days")
        print("4. Clear specific tickers")
        print("5. Optimize database")
        
        cache_choice = input("\nEnter your choice (1-5): ").strip()
        cache = get_cache()
        
        if cache_choice == '1':
            print("\n📊 Cache Statistics:")
            print("=" * 60)
            stats = cache.get_cache_stats()
            if stats.empty:
                print("No data cached yet.")
            else:
                for _, row in stats.iterrows():
                    print(f"\n{row['ticker']}:")
                    print(f"  Date range: {row['earliest_date']} to {row['latest_date']}")
                    print(f"  Records: {row['record_count']}")
                    print(f"  Last updated: {row['last_updated']}")
                print(f"\nTotal tickers cached: {len(stats)}")
        
        elif cache_choice == '2':
            confirm = input("⚠️ Clear ALL cached data? (yes/no): ").strip().lower()
            if confirm == 'yes':
                cache.clear_cache()
            else:
                print("Cancelled.")
        
        elif cache_choice == '3':
            cache.clear_cache(older_than_days=30)
        
        elif cache_choice == '4':
            tickers_str = input("Enter ticker symbols (comma-separated): ").strip()
            tickers = [t.strip().upper() for t in tickers_str.split(',')]
            confirm = input(f"Clear cache for {', '.join(tickers)}? (yes/no): ").strip().lower()
            if confirm == 'yes':
                cache.clear_cache(tickers=tickers)
            else:
                print("Cancelled.")
        
        elif cache_choice == '5':
            print("\n🔧 Optimizing database...")
            cache.optimize()
        
        else:
            print("Invalid choice.")
    else:
        print("Invalid choice. Exiting.")