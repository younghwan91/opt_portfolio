import yfinance as yf
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from data_cache import get_cache

def get_performance(tickers, end_date_str, use_cache=True):
    """
    Calculates the periodic performance for a list of specified tickers as of a given date.
    Now with smart DuckDB caching to avoid redundant data downloads!

    Args:
        tickers (list): A list of ticker symbols for which to calculate performance.
        end_date_str (str): The reference date for the performance calculation (in YYYY-MM-DD format).
        use_cache (bool): Whether to use cached data (default: True). Set to False to force fresh download.

    Returns:
        pandas.DataFrame: A DataFrame containing the periodic returns for each ticker.
    """
    # --- 1. Setup ---
    periods_in_months = [1, 3, 6, 12]
    end_date = pd.to_datetime(end_date_str)
    # Download data for the past 13 months to have a buffer.
    start_date = end_date - relativedelta(months=max(periods_in_months) + 1)

    # --- 2. Download Data (with Smart Caching) ---
    try:
        if use_cache:
            # Use smart cache - only downloads missing data
            cache = get_cache()
            data = cache.get_incremental_data(tickers, start_date, end_date)
        else:
            # Force fresh download without caching
            print("🔄 Forcing fresh download (cache bypassed)")
            data = yf.download(tickers, start=start_date, end=end_date + pd.Timedelta(days=1), auto_adjust=True)
            data = data['Close']
        
        if data.empty:
            print("Could not download data. Please check the tickers or dates.")
            return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return pd.DataFrame()

    # --- 3. Calculate Performance ---
    performance_data = {}

    for ticker in tickers:
        # Handle single vs. multiple tickers
        if len(tickers) > 1:
            ticker_prices = data[ticker].dropna()
        else:
            ticker_prices = data.dropna()

        if ticker_prices.empty:
            continue

        # Use the actual latest date from the data as the final reference date.
        actual_end_date = ticker_prices.index.max()
        end_price = ticker_prices.loc[actual_end_date]
        
        returns = {}
        for months in periods_in_months:
            # Calculate the start date for each period.
            start_date_period = actual_end_date - relativedelta(months=months)
            
            # Find the price on the closest trading day on or before the start date.
            try:
                # asof finds the last available data point on or before the specified date.
                start_price = ticker_prices.asof(start_date_period)
                
                if pd.notna(start_price) and start_price > 0:
                    period_return = ((end_price / start_price) - 1) * 100
                    returns[f'{months}-Month'] = round(period_return, 2)
                else:
                    returns[f'{months}-Month'] = None # Calculation not possible

            except KeyError:
                returns[f'{months}-Month'] = None

        performance_data[ticker] = returns
        
    # --- 4. Create Result DataFrame ---
    df_performance = pd.DataFrame(performance_data).T # Transpose rows/columns with .T
    
    # Organize column order.
    if not df_performance.empty:
        df_performance = df_performance[[f'{p}-Month' for p in periods_in_months]]
    
    return df_performance

def calculate_and_display_momentum(df, group_name):
    """Calculates momentum and displays the results for a given DataFrame."""
    print(f"--- {group_name} ---")
    if df.empty:
        print(f"No data for {group_name}.\n")
        return df

    # Ensure all return columns are numeric, coercing errors to NaN
    for col in ['1-Month', '3-Month', '6-Month', '12-Month']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in any of the period columns before calculating momentum
    df.dropna(subset=['1-Month', '3-Month', '6-Month', '12-Month'], inplace=True)
    
    if df.empty:
        print(f"Not enough data to calculate momentum for {group_name}.\n")
        return df
        
    # Calculate the momentum score.
    df['Momentum Score'] = (
        df['1-Month'] * 12 +
        df['3-Month'] * 4 +
        df['6-Month'] * 2 +
        df['12-Month'] * 1
    )

    # Sort by the momentum score.
    momentum_ranking = df.sort_values(by='Momentum Score', ascending=False)

    # Print results.
    print(momentum_ranking.to_string(
        float_format=lambda x: f'{x:.2f}',
        formatters={
            '1-Month': '{:,.2f}%'.format,
            '3-Month': '{:,.2f}%'.format,
            '6-Month': '{:,.2f}%'.format,
            '12-Month': '{:,.2f}%'.format
        }
    ))
    print("\n" + "="*50 + "\n")
    return momentum_ranking

# --- Script Execution ---
if __name__ == "__main__":
    # Define asset groups.
    aggressive_tickers = ['SPY', 'EFA', 'EEM', 'AGG']
    protective_tickers = ['LQD', 'IEF', 'SHY']

    # Set the reference date.
    calculation_date = date.today()

    print(f"ETF Performance and Momentum as of '{calculation_date}' 📈\n")

    # --- Process Aggressive Assets ---
    agg_performance = get_performance(aggressive_tickers, calculation_date)
    agg_momentum = calculate_and_display_momentum(agg_performance, "Aggressive Assets")

    # --- Process Protective Assets ---
    prot_performance = get_performance(protective_tickers, calculation_date)
    prot_momentum = calculate_and_display_momentum(prot_performance, "Protective Assets")

    # --- Investment Decision Logic ---
    print("--- Investment Decision ---")

    if not agg_momentum.empty:
        # Check if ANY aggressive asset has a momentum score less than 0.
        is_any_aggressive_negative = (agg_momentum['Momentum Score'] < 0).any()

        if is_any_aggressive_negative:
            print("At least one aggressive asset has negative momentum. Selecting from protective assets.")
            if not prot_momentum.empty:
                selection = prot_momentum.index[0]
                selection_score = prot_momentum['Momentum Score'].iloc[0]
                print(f"👉 Selected Protective Asset: {selection} (Momentum Score: {selection_score:.2f})")
            else:
                print("Cannot make a selection; no data for protective assets.")
        else:
            print("All aggressive assets have positive momentum. Selecting from aggressive assets.")
            selection = agg_momentum.index[0]
            selection_score = agg_momentum['Momentum Score'].iloc[0]
            print(f"👉 Selected Aggressive Asset: {selection} (Momentum Score: {selection_score:.2f})")
    else:
        print("Cannot make a decision; no data for aggressive assets.")