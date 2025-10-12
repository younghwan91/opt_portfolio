import subprocess
import sys

# Run when __name__ == "__main__"
if __name__ == "__main__":
    print("🚀 PORTFOLIO MANAGEMENT SYSTEM")
    print("=" * 40)
    print("Choose an option:")
    print("1. Run integrated portfolio management (CLI)")
    print("2. Launch web UI (Streamlit)")
    print("3. Run VAA analysis only")
    print("4. Run portfolio calculator only")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
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
    else:
        print("Invalid choice. Exiting.")