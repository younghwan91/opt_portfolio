import subprocess
import sys
from datetime import date
from vaa_agg import get_performance, calculate_and_display_momentum
from port_ratio_calculator import calculate_capital_ratios
from rebalance import calculate_rebalance, print_rebalance_report

def run_vaa_selection():
    """Run VAA aggregation and return the selected ETF."""
    aggressive_tickers = ['SPY', 'EFA', 'EEM', 'AGG']
    protective_tickers = ['LQD', 'IEF', 'SHY']
    calculation_date = date.today()

    print(f"ETF Performance and Momentum as of '{calculation_date}' 📈\n")

    # Process Aggressive Assets
    agg_performance = get_performance(aggressive_tickers, calculation_date)
    agg_momentum = calculate_and_display_momentum(agg_performance, "Aggressive Assets")

    # Process Protective Assets
    prot_performance = get_performance(protective_tickers, calculation_date)
    prot_momentum = calculate_and_display_momentum(prot_performance, "Protective Assets")

    # Investment Decision Logic
    print("--- Investment Decision ---")
    selected_etf = None

    if not agg_momentum.empty:
        is_any_aggressive_negative = (agg_momentum['Momentum Score'] < 0).any()

        if is_any_aggressive_negative:
            print("At least one aggressive asset has negative momentum. Selecting from protective assets.")
            if not prot_momentum.empty:
                selected_etf = prot_momentum.index[0]
                selection_score = prot_momentum['Momentum Score'].iloc[0]
                print(f"👉 Selected Protective Asset: {selected_etf} (Momentum Score: {selection_score:.2f})")
            else:
                print("Cannot make a selection; no data for protective assets.")
        else:
            print("All aggressive assets have positive momentum. Selecting from aggressive assets.")
            selected_etf = agg_momentum.index[0]
            selection_score = agg_momentum['Momentum Score'].iloc[0]
            print(f"👉 Selected Aggressive Asset: {selected_etf} (Momentum Score: {selection_score:.2f})")
    
    return selected_etf

def main_portfolio_management():
    """Main portfolio management function."""
    print("🚀 INTEGRATED PORTFOLIO MANAGEMENT SYSTEM")
    print("="*50)
    
    # Step 1: Run VAA to select ETF
    print("\n📊 STEP 1: VAA ETF SELECTION")
    print("-" * 30)
    selected_etf = run_vaa_selection()
    
    if not selected_etf:
        print("❌ Could not select an ETF. Exiting.")
        return
    
    # Step 2: Get current portfolio from user input
    print(f"\n💼 STEP 2: CURRENT PORTFOLIO INPUT")
    print("-" * 40)
    
    # Get user input for current portfolio
    current_portfolio = {}
    print(f"📊 Target Allocation Strategy:")
    print(f"   🎯 {selected_etf}: 50%")
    print(f"   📈 SPY: 12.5%")
    print(f"   🏛️ TLT: 12.5%") 
    print(f"   🥇 GLD: 12.5%")
    print(f"   💵 BIL: 12.5%")
    print()
    
    # Selected ETF first (prominent)
    print(f"Enter your current portfolio holdings:")
    print(f"⭐ Selected ETF (50% target allocation):")
    while True:
        try:
            shares = int(input(f"   {selected_etf} shares: "))
            if shares >= 0:
                current_portfolio[selected_etf] = shares
                break
            else:
                print("   Please enter a non-negative number.")
        except ValueError:
            print("   Please enter a valid number.")
    
    print(f"\n📊 Core Holdings (12.5% each):")
    core_etfs = ['SPY', 'TLT', 'GLD', 'BIL']
    for etf in core_etfs:
        while True:
            try:
                shares = int(input(f"   {etf} shares: "))
                if shares >= 0:
                    current_portfolio[etf] = shares
                    break
                else:
                    print("   Please enter a non-negative number.")
            except ValueError:
                print("   Please enter a valid number.")
    
    print(f"\n📊 CURRENT PORTFOLIO ANALYSIS")
    print("-" * 30)
    portfolio_breakdown = calculate_capital_ratios(current_portfolio)
    if not portfolio_breakdown.empty:
        print(portfolio_breakdown.to_string(
            formatters={
                'Shares': '{:,.0f}'.format,
                'Current Price': '${:,.2f}'.format,
                'Market Value': '${:,.2f}'.format,
                'Capital Ratio (%)': '{:.2f}%'.format
            }
        ))
    
    # Step 3: Interactive rebalancing
    print(f"\n⚖️  STEP 3: PORTFOLIO REBALANCING")
    print("-" * 35)
    
    while True:
        print(f"\nSelected ETF: {selected_etf}")
        print("\nChoose an option:")
        print("1. Rebalance with current holdings only")
        print("2. Add additional cash and rebalance")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🔄 Calculating optimal rebalancing...")
            recommendations = calculate_rebalance(current_portfolio, selected_etf, 0)
            print_rebalance_report(recommendations)
            
            # Ask if user wants to see alternative scenarios
            print(f"\n❓ Would you like to see what happens with additional cash? (y/n): ", end="")
            response = input().strip().lower()
            if response == 'y':
                try:
                    additional_cash = float(input("Enter additional cash amount: $"))
                    print(f"\n🔄 Calculating with ${additional_cash:,.2f} additional cash...")
                    alt_recommendations = calculate_rebalance(current_portfolio, selected_etf, additional_cash)
                    print_rebalance_report(alt_recommendations)
                except ValueError:
                    print("❌ Invalid amount entered.")
            break
            
        elif choice == '2':
            try:
                additional_cash = float(input("Enter additional cash amount: $"))
                print(f"\n🔄 Calculating optimal rebalancing with ${additional_cash:,.2f} additional cash...")
                recommendations = calculate_rebalance(current_portfolio, selected_etf, additional_cash)
                print_rebalance_report(recommendations)
                break
            except ValueError:
                print("❌ Please enter a valid number.")
                
        elif choice == '3':
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main_portfolio_management()