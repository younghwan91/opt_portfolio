import subprocess
import sys
from data_cache import get_cache

# Run when __name__ == "__main__"
if __name__ == "__main__":
    print("🚀 PORTFOLIO MANAGEMENT SYSTEM")
    print("=" * 40)
    print("Choose an option:")
    print("1. Run integrated portfolio management (CLI)")
    print("2. Launch web UI (Streamlit)")
    print("3. Run VAA analysis only")
    print("4. Run portfolio calculator only")
    print("5. Cache management")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
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