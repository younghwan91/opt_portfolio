import yfinance as yf
import pandas as pd

def calculate_capital_ratios(portfolio):
    """
    Calculates the capital allocation ratio for a given portfolio of tickers and shares.

    Args:
        portfolio (dict): A dictionary with ticker symbols as keys and number of shares as values.
                          Example: {'SPY': 10, 'EEM': 20}

    Returns:
        pandas.DataFrame: A DataFrame with the portfolio breakdown including capital ratios,
                          or an empty DataFrame if data cannot be fetched.
    """
    if not portfolio:
        print("The portfolio dictionary is empty. Please provide tickers and shares.")
        return pd.DataFrame()

    tickers = list(portfolio.keys())
    print(f"Fetching current prices for: {', '.join(tickers)}...")

    try:
        # yf.Ticker(...).info is more reliable for the latest price ('regularMarketPrice')
        # We will fetch the 'regularMarketPrice' for each ticker.
        price_data = {}
        for ticker in tickers:
            t = yf.Ticker(ticker)
            info = t.info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                price_data[ticker] = info['regularMarketPrice']
            else:
                # Fallback to the last closing price if regularMarketPrice is not available
                hist = t.history(period="1d")
                if not hist.empty:
                    price_data[ticker] = hist['Close'].iloc[-1]
                else:
                    print(f"Warning: Could not fetch price for {ticker}. It will be excluded.")
                    continue
        
        if not price_data:
            print("Could not fetch price data for any of the tickers.")
            return pd.DataFrame()

    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return pd.DataFrame()

    # --- Create a DataFrame for analysis ---
    df = pd.DataFrame.from_dict(portfolio, orient='index', columns=['Shares'])
    df['Current Price'] = df.index.map(price_data)

    # --- Calculate Market Value and Ratios ---
    df.dropna(subset=['Current Price'], inplace=True) # Remove tickers for which price was not found
    
    if df.empty:
        print("No valid price data to perform calculations.")
        return pd.DataFrame()

    df['Market Value'] = df['Shares'] * df['Current Price']
    
    total_portfolio_value = df['Market Value'].sum()
    
    if total_portfolio_value > 0:
        df['Capital Ratio (%)'] = (df['Market Value'] / total_portfolio_value) * 100
    else:
        df['Capital Ratio (%)'] = 0

    return df

# --- Script Execution ---
if __name__ == "__main__":
    print("📊 PORTFOLIO CAPITAL RATIO CALCULATOR")
    print("=" * 40)
    print("❌ ERROR: Cannot calculate ratios without knowing your VAA selection!")
    print()
    print("Please run one of these instead:")
    print("1. python main.py (choose option 4)")
    print("2. python integrated_portfolio.py (complete analysis)")
    print("3. streamlit run portfolio_ui.py (web interface)")
    print()
    print("💡 The system needs to first determine which ETF gets 50% allocation")
    print("   through VAA analysis before calculating meaningful ratios.")
