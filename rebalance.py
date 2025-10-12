import yfinance as yf
import pandas as pd
from port_ratio_calculator import calculate_capital_ratios
import math

def get_current_prices(tickers):
    """Get current prices for a list of tickers."""
    price_data = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                price_data[ticker] = info['regularMarketPrice']
            else:
                hist = t.history(period="1d")
                if not hist.empty:
                    price_data[ticker] = hist['Close'].iloc[-1]
        except:
            print(f"Warning: Could not fetch price for {ticker}")
    return price_data

def calculate_rebalance(current_portfolio, selected_etf, additional_cash=0):
    """
    Calculate optimized rebalancing for the portfolio with error analysis.
    
    Args:
        current_portfolio (dict): Current holdings {ticker: shares}
        selected_etf (str): The ETF selected by VAA (gets 50% allocation)
        additional_cash (float): Additional cash to invest (default: 0)
    
    Returns:
        dict: Rebalancing recommendations with optimization and error analysis
    """
    # Define permanent ETFs (each gets 12.5%)
    permanent_etfs = ['SPY', 'TLT', 'GLD', 'BIL']
    all_etfs = [selected_etf] + permanent_etfs
    
    # Get current prices
    all_tickers = list(current_portfolio.keys())
    for etf in all_etfs:
        if etf not in all_tickers:
            all_tickers.append(etf)
    
    prices = get_current_prices(all_tickers)
    
    if not prices:
        return {"error": "Could not fetch price data"}
    
    # Calculate current portfolio value (only for sellable assets)
    current_value = 0
    sellable_value = 0
    for ticker, shares in current_portfolio.items():
        if ticker in prices and shares > 0:
            value = shares * prices[ticker]
            current_value += value
            sellable_value += value
    
    # Calculate available cash for rebalancing
    available_cash = additional_cash
    
    # First, check what we can sell to generate more cash
    current_allocations = {}
    for ticker, shares in current_portfolio.items():
        if ticker in prices:
            current_allocations[ticker] = shares * prices[ticker]
        else:
            current_allocations[ticker] = 0
    
    # Total target value with additional cash
    total_target_value = current_value + additional_cash
    
    # Ideal target allocations
    ideal_targets = {}
    ideal_targets[selected_etf] = total_target_value * 0.50  # 50%
    for etf in permanent_etfs:
        ideal_targets[etf] = total_target_value * 0.125  # 12.5% each
    
    # Optimize allocation considering cash constraints
    optimized_portfolio = optimize_portfolio_allocation(
        current_portfolio, ideal_targets, prices, available_cash
    )
    
    return optimized_portfolio

def optimize_portfolio_allocation(current_portfolio, ideal_targets, prices, additional_cash):
    """
    Optimize portfolio allocation with cash and fractional share constraints.
    """
    # Initialize results
    recommendations = {
        'current_value': 0,
        'additional_cash': additional_cash,
        'total_target_value': 0,
        'selected_etf': None,
        'transactions': {},
        'optimized_portfolio': {},
        'allocation_errors': {},
        'remaining_cash': additional_cash,
        'optimization_summary': {}
    }
    
    # Calculate current values
    current_values = {}
    total_current_value = 0
    for ticker, shares in current_portfolio.items():
        if ticker in prices:
            value = shares * prices[ticker]
            current_values[ticker] = value
            total_current_value += value
    
    recommendations['current_value'] = total_current_value
    recommendations['total_target_value'] = total_current_value + additional_cash
    
    # Find selected ETF (the one with 50% target)
    for ticker, target_value in ideal_targets.items():
        if abs(target_value / recommendations['total_target_value'] - 0.5) < 0.01:
            recommendations['selected_etf'] = ticker
            break
    
    # Phase 1: Calculate what we can sell to raise cash
    cash_from_sales = 0
    sales_transactions = {}
    
    for ticker in current_portfolio.keys():
        if ticker in prices:
            current_shares = current_portfolio.get(ticker, 0)
            current_value = current_shares * prices[ticker]
            target_value = ideal_targets.get(ticker, 0)
            
            if current_value > target_value and current_shares > 0:
                # We have excess - calculate how much to sell
                excess_value = current_value - target_value
                shares_to_sell = math.floor(excess_value / prices[ticker])
                
                if shares_to_sell > 0:
                    sale_proceeds = shares_to_sell * prices[ticker]
                    cash_from_sales += sale_proceeds
                    
                    sales_transactions[ticker] = {
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'proceeds': sale_proceeds,
                        'price': prices[ticker]
                    }
    
    # Total available cash
    total_available_cash = additional_cash + cash_from_sales
    recommendations['total_available_cash'] = total_available_cash
    
    # Phase 2: Optimize purchases with available cash
    remaining_cash = total_available_cash
    purchase_transactions = {}
    final_portfolio = current_portfolio.copy()
    
    # Apply sales first
    for ticker, transaction in sales_transactions.items():
        final_portfolio[ticker] = final_portfolio.get(ticker, 0) - transaction['shares']
    
    # Calculate shortfalls and prioritize purchases
    shortfalls = []
    for ticker, target_value in ideal_targets.items():
        if ticker in prices:
            current_shares_after_sales = final_portfolio.get(ticker, 0)
            current_value_after_sales = current_shares_after_sales * prices[ticker]
            shortfall = target_value - current_value_after_sales
            
            if shortfall > 0:
                max_shares_can_buy = math.floor(shortfall / prices[ticker])
                shortfalls.append({
                    'ticker': ticker,
                    'shortfall_value': shortfall,
                    'max_shares': max_shares_can_buy,
                    'cost_per_share': prices[ticker],
                    'priority': shortfall  # Can adjust priority logic here
                })
    
    # Sort by priority (largest shortfall first for now)
    shortfalls.sort(key=lambda x: x['priority'], reverse=True)
    
    # Execute purchases within cash constraints
    for item in shortfalls:
        ticker = item['ticker']
        max_affordable_shares = math.floor(remaining_cash / item['cost_per_share'])
        shares_to_buy = min(item['max_shares'], max_affordable_shares)
        
        if shares_to_buy > 0:
            cost = shares_to_buy * item['cost_per_share']
            remaining_cash -= cost
            
            final_portfolio[ticker] = final_portfolio.get(ticker, 0) + shares_to_buy
            purchase_transactions[ticker] = {
                'action': 'BUY',
                'shares': shares_to_buy,
                'cost': cost,
                'price': item['cost_per_share']
            }
    
    # Combine all transactions
    all_transactions = {**sales_transactions, **purchase_transactions}
    
    # Calculate final allocations and errors
    final_values = {}
    total_final_value = 0
    allocation_errors = {}
    
    for ticker, shares in final_portfolio.items():
        if ticker in prices and shares > 0:
            value = shares * prices[ticker]
            final_values[ticker] = value
            total_final_value += value
    
    # Calculate allocation errors
    for ticker, target_value in ideal_targets.items():
        actual_value = final_values.get(ticker, 0)
        target_percentage = (target_value / recommendations['total_target_value']) * 100
        actual_percentage = (actual_value / total_final_value * 100) if total_final_value > 0 else 0
        
        allocation_errors[ticker] = {
            'target_value': target_value,
            'actual_value': actual_value,
            'target_percentage': target_percentage,
            'actual_percentage': actual_percentage,
            'value_error': actual_value - target_value,
            'percentage_error': actual_percentage - target_percentage
        }
    
    # Populate final recommendations
    recommendations['transactions'] = all_transactions
    recommendations['optimized_portfolio'] = final_portfolio
    recommendations['allocation_errors'] = allocation_errors
    recommendations['remaining_cash'] = remaining_cash
    recommendations['final_portfolio_value'] = total_final_value
    recommendations['cash_from_sales'] = cash_from_sales
    
    # Add optimization summary
    recommendations['optimization_summary'] = {
        'total_transactions': len(all_transactions),
        'total_sales_proceeds': cash_from_sales,
        'total_purchase_cost': total_available_cash - remaining_cash,
        'cash_utilization_rate': ((total_available_cash - remaining_cash) / total_available_cash * 100) if total_available_cash > 0 else 0
    }
    
    return recommendations

def print_rebalance_report(recommendations):
    """Print a formatted rebalancing report with optimization details."""
    if 'error' in recommendations:
        print(f"❌ Error: {recommendations['error']}")
        return
    
    print("=" * 70)
    print("📊 OPTIMIZED PORTFOLIO REBALANCING REPORT")
    print("=" * 70)
    
    # Portfolio Summary
    print(f"🎯 Selected ETF (VAA): {recommendations['selected_etf']}")
    print(f"💰 Current Portfolio Value: ${recommendations['current_value']:,.2f}")
    print(f"💵 Additional Cash Available: ${recommendations['additional_cash']:,.2f}")
    print(f"🎯 Target Portfolio Value: ${recommendations['total_target_value']:,.2f}")
    if 'cash_from_sales' in recommendations:
        print(f"💸 Cash from Sales: ${recommendations['cash_from_sales']:,.2f}")
        print(f"💳 Total Available Cash: ${recommendations.get('total_available_cash', 0):,.2f}")
    print()
    
    # Transactions Section
    print("📋 REQUIRED TRANSACTIONS:")
    print("-" * 50)
    
    if not recommendations['transactions']:
        print("✅ No transactions needed - portfolio is optimally balanced!")
    else:
        sales_total = 0
        purchases_total = 0
        
        for ticker, trans in recommendations['transactions'].items():
            if trans['action'] == 'SELL':
                print(f"🔴 SELL {ticker}: {trans['shares']} shares @ ${trans['price']:.2f} = +${trans['proceeds']:,.2f}")
                sales_total += trans['proceeds']
            elif trans['action'] == 'BUY':
                print(f"🟢 BUY  {ticker}: {trans['shares']} shares @ ${trans['price']:.2f} = -${trans['cost']:,.2f}")
                purchases_total += trans['cost']
        
        print(f"\n💰 Total Sales Proceeds: +${sales_total:,.2f}")
        print(f"💸 Total Purchase Cost: -${purchases_total:,.2f}")
        print(f"💵 Net Cash Flow: ${sales_total - purchases_total:,.2f}")
    
    print()
    
    # Optimized Portfolio vs Target
    print("🎯 OPTIMIZED PORTFOLIO vs TARGET ALLOCATION:")
    print("-" * 65)
    print(f"{'ETF':<6} {'Shares':<8} {'Value':<12} {'Target%':<9} {'Actual%':<9} {'Error':<10}")
    print("-" * 65)
    
    total_error_abs = 0
    
    for ticker in recommendations['allocation_errors'].keys():
        error_data = recommendations['allocation_errors'][ticker]
        shares = recommendations['optimized_portfolio'].get(ticker, 0)
        
        target_pct = error_data['target_percentage']
        actual_pct = error_data['actual_percentage']
        pct_error = error_data['percentage_error']
        
        total_error_abs += abs(pct_error)
        
        # Color coding for errors
        error_symbol = "✅" if abs(pct_error) < 1.0 else "⚠️" if abs(pct_error) < 3.0 else "❌"
        
        print(f"{ticker:<6} {shares:<8.0f} ${error_data['actual_value']:<11,.0f} {target_pct:<8.1f}% {actual_pct:<8.1f}% {pct_error:+5.1f}% {error_symbol}")
    
    print("-" * 65)
    print(f"📊 Average Allocation Error: {total_error_abs / len(recommendations['allocation_errors']):.2f}%")
    
    # Cash and Optimization Summary
    print(f"\n💰 CASH SUMMARY:")
    print("-" * 30)
    print(f"Remaining Cash: ${recommendations['remaining_cash']:,.2f}")
    
    if 'optimization_summary' in recommendations:
        opt_summary = recommendations['optimization_summary']
        print(f"Cash Utilization: {opt_summary['cash_utilization_rate']:.1f}%")
        print(f"Total Transactions: {opt_summary['total_transactions']}")
    
    print(f"Final Portfolio Value: ${recommendations.get('final_portfolio_value', 0):,.2f}")
    
    # Optimization Quality Assessment
    print(f"\n🎯 OPTIMIZATION QUALITY:")
    print("-" * 30)
    avg_error = total_error_abs / len(recommendations['allocation_errors'])
    if avg_error < 1.0:
        quality = "🟢 Excellent (< 1% average error)"
    elif avg_error < 2.5:
        quality = "🟡 Good (< 2.5% average error)"
    elif avg_error < 5.0:
        quality = "🟠 Fair (< 5% average error)"
    else:
        quality = "🔴 Needs Improvement (> 5% average error)"
    
    print(f"Allocation Quality: {quality}")
    
    if avg_error > 2.0:
        print(f"💡 Tip: Consider adding more cash or selling additional positions to improve allocation accuracy.")
    
    print("=" * 70)

# Script execution
if __name__ == "__main__":
    print("🔧 PORTFOLIO REBALANCING CALCULATOR")
    print("=" * 40)
    print("❌ ERROR: Cannot run rebalancing without VAA selection!")
    print("Please run one of these instead:")
    print("1. python main.py (full menu)")
    print("2. python integrated_portfolio.py (complete CLI)")
    print("3. streamlit run portfolio_ui.py (web interface)")
    print()
    print("💡 The system needs to first determine which ETF gets 50% allocation")
    print("   through VAA analysis before calculating meaningful rebalancing.")