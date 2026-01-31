import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
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

def calculate_historical_momentum(tickers, start_date, end_date, use_cache=True):
    """
    Calculates historical momentum scores for a list of tickers over a date range.
    
    Args:
        tickers (list): List of ticker symbols.
        start_date (str/date): Start date for the analysis.
        end_date (str/date): End date for the analysis.
        use_cache (bool): Whether to use cached data.
        
    Returns:
        pandas.DataFrame: DataFrame with dates as index and momentum scores for each ticker as columns.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # We need extra data before start_date to calculate the first momentum score
    # 12 months ~ 365 days + buffer
    fetch_start = start_dt - timedelta(days=400)
    
    print(f"📊 Calculating historical momentum from {start_date} to {end_date}...")
    
    try:
        if use_cache:
            cache = get_cache()
            data = cache.get_incremental_data(tickers, fetch_start, end_dt)
        else:
            data = yf.download(tickers, start=fetch_start, end=end_dt + timedelta(days=1), auto_adjust=True)['Close']
            
        if data.empty:
            return pd.DataFrame()
            
        # Calculate rolling returns (approximate months to trading days)
        # 1m=21, 3m=63, 6m=126, 12m=252
        r1 = data.pct_change(21) * 100
        r3 = data.pct_change(63) * 100
        r6 = data.pct_change(126) * 100
        r12 = data.pct_change(252) * 100
        
        # Calculate Momentum Score
        # Formula: 12*r1 + 4*r3 + 2*r6 + 1*r12
        momentum_scores = (r1 * 12) + (r3 * 4) + (r6 * 2) + (r12 * 1)
        
        # Filter for the requested date range
        momentum_scores = momentum_scores.loc[start_dt:end_dt]
        
        return momentum_scores.dropna()
        
    except Exception as e:
        print(f"Error calculating historical momentum: {e}")
        return pd.DataFrame()

def simulate_momentum_ou(momentum_df, months=1, num_simulations=1000):
    """
    Simulates future momentum scores using an Ornstein-Uhlenbeck (OU) process.
    Returns the win probability for each asset and the mean forecast path.
    
    Args:
        momentum_df (pandas.DataFrame): Historical momentum scores.
        months (int): Number of months to simulate.
        num_simulations (int): Number of Monte Carlo paths to simulate.
        
    Returns:
        tuple: (win_probabilities (pd.Series), forecast_df (pd.DataFrame))
    """
    if momentum_df.empty:
        return pd.Series(), pd.DataFrame()
        
    print(f"🎲 Simulating momentum (OU Process) for next {months} months ({num_simulations} paths)...")
    
    forecast_days = months * 21
    dt = 1.0  # Time step (1 day)
    
    # Results containers
    final_scores = {ticker: [] for ticker in momentum_df.columns}
    mean_paths = {}
    
    # Create future dates
    last_date = momentum_df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
    
    for ticker in momentum_df.columns:
        # Get historical series
        series = momentum_df[ticker].dropna()
        if len(series) < 30:
            print(f"⚠️ Not enough data for {ticker}, skipping simulation.")
            continue
            
        # Calibrate OU parameters (Theta, Mu, Sigma)
        # dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        # Discrete: X_{t+1} = X_t + theta*(mu - X_t) + sigma*epsilon
        
        # Regress X_{t+1} on X_t to find parameters
        x_t = series.values[:-1]
        x_tp1 = series.values[1:]
        
        # Linear regression: x_{t+1} = alpha + beta * x_t + error
        # beta = 1 - theta
        # alpha = theta * mu
        # sigma = std(error)
        
        slope, intercept = np.polyfit(x_t, x_tp1, 1)
        residuals = x_tp1 - (slope * x_t + intercept)
        
        theta = 1 - slope
        mu = intercept / theta if abs(theta) > 1e-6 else series.mean()
        sigma = np.std(residuals)
        
        # Constrain theta to ensure stability (0 < theta < 1)
        theta = max(min(theta, 0.1), 0.001) 
        
        # Simulation
        current_val = series.iloc[-1]
        sim_paths = np.zeros((num_simulations, forecast_days))
        sim_paths[:, 0] = current_val
        
        for t in range(1, forecast_days):
            # Vectorized update for all paths
            noise = np.random.normal(0, 1, num_simulations)
            # OU Update
            dx = theta * (mu - sim_paths[:, t-1]) * dt + sigma * noise
            sim_paths[:, t] = sim_paths[:, t-1] + dx
            
        # Store final values for probability calculation
        final_scores[ticker] = sim_paths[:, -1]
        
        # Store mean path for plotting
        mean_paths[ticker] = np.mean(sim_paths, axis=0)

    # Calculate Win Probabilities
    # For each simulation run, see which asset ended highest
    wins = {ticker: 0 for ticker in momentum_df.columns}
    valid_tickers = [t for t in momentum_df.columns if t in final_scores and len(final_scores[t]) > 0]
    
    if not valid_tickers:
        return pd.Series(), pd.DataFrame()

    # Convert final scores to array: [num_sims, num_tickers]
    all_final_scores = np.array([final_scores[t] for t in valid_tickers]).T
    
    # Find winner for each simulation
    winners_indices = np.argmax(all_final_scores, axis=1)
    
    for idx in winners_indices:
        winner_ticker = valid_tickers[idx]
        wins[winner_ticker] += 1
        
    probs = pd.Series(wins) / num_simulations
    probs = probs.sort_values(ascending=False)
    
    # Create forecast DataFrame
    df_forecast = pd.DataFrame(mean_paths, index=future_dates)
    
    return probs, df_forecast

def plot_momentum_history(momentum_df, title="VAA Momentum History", forecast_df=None, win_probs=None):
    """
    Plots historical momentum and a probability bar chart for the forecast.
    """
    if momentum_df.empty:
        print("No data to plot.")
        return

    # Create a figure with 2 subplots (Time Series & Bar Chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # --- Plot 1: Time Series ---
    colors = plt.cm.tab10(range(len(momentum_df.columns)))
    color_map = {col: colors[i] for i, col in enumerate(momentum_df.columns)}
    
    for column in momentum_df.columns:
        color = color_map.get(column, 'black')
        ax1.plot(momentum_df.index, momentum_df[column], label=column, color=color, linewidth=2)
        
        # Plot forecast if available
        if forecast_df is not None and column in forecast_df.columns:
            ax1.plot(forecast_df.index, forecast_df[column], linestyle='--', color=color, alpha=0.7)
            ax1.plot(forecast_df.index[-1], forecast_df[column].iloc[-1], marker='o', color=color, markersize=4)

    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Zero Line')
    ax1.axvline(x=momentum_df.index[-1], color='gray', linestyle=':', alpha=0.5, label='Today')
    
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Momentum Score')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Win Probabilities ---
    if win_probs is not None and not win_probs.empty:
        y_pos = np.arange(len(win_probs))
        # Match colors to the time series
        bar_colors = [color_map.get(t, 'gray') for t in win_probs.index]
        
        ax2.barh(y_pos, win_probs.values * 100, align='center', color=bar_colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(win_probs.index)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_xlabel('Probability of Being Best (%)')
        ax2.set_title('Next Month "Best Asset" Probability')
        
        # Add percentage labels
        for i, v in enumerate(win_probs.values):
            ax2.text(v * 100 + 1, i, f"{v*100:.1f}%", va='center')
            
        ax2.set_xlim(0, 115) # Make room for text
    else:
        ax2.text(0.5, 0.5, "No Probability Data", ha='center', va='center')
        ax2.axis('off')

    plt.tight_layout()
    print("📈 Displaying plot with OU Forecast...")
    plt.show()

def calculate_ou_forecast(series, months=1):
    """
    Calibrates OU parameters and returns the expected value 'months' ahead.
    Uses analytical solution: E[X_{t+T}] = mu + (X_t - mu) * exp(-theta * T)
    """
    if len(series) < 30:
        return series.iloc[-1] # Fallback to current value

    # Regress X_{t+1} on X_t
    x_t = series.values[:-1]
    x_tp1 = series.values[1:]
    
    # Linear regression: x_{t+1} = alpha + beta * x_t
    slope, intercept = np.polyfit(x_t, x_tp1, 1)
    
    # Constrain slope to be stable (0 < slope < 1) for mean reversion
    slope = max(min(slope, 0.999), 0.001)
    
    theta = -np.log(slope)
    mu = intercept / (1 - slope)
    
    # Forecast
    current_val = series.iloc[-1]
    T = months * 21 # Trading days
    
    expected_val = mu + (current_val - mu) * np.exp(-theta * T)
    return expected_val

def analyze_strategies(momentum_df):
    """
    Analyzes the momentum data using multiple strategies and returns the best asset for each.
    """
    if momentum_df.empty:
        return {}
        
    recommendations = {}
    
    # 1. Standard VAA (Current Score)
    current_scores = momentum_df.iloc[-1]
    recommendations['Standard (Current)'] = {
        'Asset': current_scores.idxmax(),
        'Score': current_scores.max()
    }
    
    # Calculate forecasts
    forecasts_1m = {}
    forecasts_3m = {}
    forecasts_6m = {}
    deltas = {}
    
    for ticker in momentum_df.columns:
        series = momentum_df[ticker].dropna()
        
        # Use the last 60 days for calibration (consistent with backtest)
        recent_series = series.tail(60)
        
        f1 = calculate_ou_forecast(recent_series, months=1)
        f3 = calculate_ou_forecast(recent_series, months=3)
        f6 = calculate_ou_forecast(recent_series, months=6)
        
        forecasts_1m[ticker] = f1
        forecasts_3m[ticker] = f3
        forecasts_6m[ticker] = f6
        deltas[ticker] = f1 - series.iloc[-1] # Velocity (1M Forecast - Current)

    # 2. Forecast 1M
    s1 = pd.Series(forecasts_1m)
    recommendations['OU Forecast (1-Month)'] = {'Asset': s1.idxmax(), 'Score': s1.max()}
    
    # 3. Forecast 3M
    s3 = pd.Series(forecasts_3m)
    recommendations['OU Forecast (3-Month)'] = {'Asset': s3.idxmax(), 'Score': s3.max()}
    
    # 4. Forecast 6M
    s6 = pd.Series(forecasts_6m)
    recommendations['OU Forecast (6-Month)'] = {'Asset': s6.idxmax(), 'Score': s6.max()}
    
    # 5. Velocity
    sd = pd.Series(deltas)
    recommendations['Momentum Velocity (Delta)'] = {'Asset': sd.idxmax(), 'Score': sd.max()}
    
    return recommendations

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
                # Calculate historical momentum for protective assets to run strategies
                # We need history for the strategies, but calculate_and_display_momentum only returns the current snapshot
                # So we need to fetch history again or refactor.
                # For now, let's fetch history for the protective group
                end_date = date.today()
                start_date = end_date - timedelta(days=365*2)
                prot_hist_mom = calculate_historical_momentum(protective_tickers, start_date, end_date)
                
                recs = analyze_strategies(prot_hist_mom)
                print("\n🏆 Best Protective Asset by Strategy:")
                for strategy, details in recs.items():
                    print(f"  • {strategy:<25}: {details['Asset']} (Score: {details['Score']:.2f})")
            else:
                print("Cannot make a selection; no data for protective assets.")
        else:
            print("All aggressive assets have positive momentum. Selecting from aggressive assets.")
            
            # Calculate historical momentum for aggressive assets
            end_date = date.today()
            start_date = end_date - timedelta(days=365*2)
            agg_hist_mom = calculate_historical_momentum(aggressive_tickers, start_date, end_date)
            
            recs = analyze_strategies(agg_hist_mom)
            print("\n🏆 Best Aggressive Asset by Strategy:")
            for strategy, details in recs.items():
                print(f"  • {strategy:<25}: {details['Asset']} (Score: {details['Score']:.2f})")
                
    else:
        print("Cannot make a decision; no data for aggressive assets.")