
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from data_cache import get_cache
import vaa_agg

def calibrate_ou_and_forecast(series, months=1):
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
    
    # beta = exp(-theta * dt) -> theta = -ln(beta) / dt
    # alpha = mu * (1 - beta) -> mu = alpha / (1 - beta)
    # Assuming dt = 1 day
    
    # Constrain slope to be stable (0 < slope < 1) for mean reversion
    slope = max(min(slope, 0.999), 0.001)
    
    theta = -np.log(slope)
    mu = intercept / (1 - slope)
    
    # Forecast
    current_val = series.iloc[-1]
    T = months * 21 # Trading days
    
    expected_val = mu + (current_val - mu) * np.exp(-theta * T)
    return expected_val

def calculate_momentum_series(hist_data, tickers):
    """
    Calculates the momentum score series for the given tickers using the provided historical data.
    Returns a DataFrame of momentum scores (last 60 days).
    """
    momentum_data = {}
    
    # We need enough data for the longest window (252 days) + buffer for rolling (60 days)
    # Total ~320 days
    recent_data = hist_data.tail(320)
    
    if len(recent_data) < 252:
        return pd.DataFrame()

    for ticker in tickers:
        if ticker not in recent_data.columns:
            continue
            
        prices = recent_data[ticker]
        
        # Calculate rolling returns
        r1 = prices.pct_change(21) * 100
        r3 = prices.pct_change(63) * 100
        r6 = prices.pct_change(126) * 100
        r12 = prices.pct_change(252) * 100
        
        # Weighted momentum
        mom = (r1 * 12) + (r3 * 4) + (r6 * 2) + (r12 * 1)
        momentum_data[ticker] = mom.dropna().tail(60) # Keep last 60 days
        
    return pd.DataFrame(momentum_data)

def select_asset(momentum_df, strategy='current'):
    """
    Selects the best asset from the momentum DataFrame based on the strategy.
    Strategies: 'current', 'forecast_1m', 'forecast_3m', 'forecast_6m', 'delta'
    """
    if momentum_df.empty:
        return None

    if strategy == 'current':
        return momentum_df.iloc[-1].idxmax()
    
    scores = {}
    current_vals = momentum_df.iloc[-1]
    
    for ticker in momentum_df.columns:
        series = momentum_df[ticker]
        
        if strategy == 'forecast_1m':
            scores[ticker] = calibrate_ou_and_forecast(series, months=1)
        elif strategy == 'forecast_3m':
            scores[ticker] = calibrate_ou_and_forecast(series, months=3)
        elif strategy == 'forecast_6m':
            scores[ticker] = calibrate_ou_and_forecast(series, months=6)
        elif strategy == 'delta':
            forecast = calibrate_ou_and_forecast(series, months=1)
            scores[ticker] = forecast - current_vals[ticker]
            
    return pd.Series(scores).idxmax()

def run_backtest():
    print("🚀 Starting Backtest Comparison (With Protective Logic)...")
    
    # Configuration
    agg_tickers = ['SPY', 'EFA', 'EEM', 'AGG']
    prot_tickers = ['LQD', 'IEF', 'SHY']
    all_tickers = list(set(agg_tickers + prot_tickers))
    
    backtest_years = 15
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=backtest_years)
    
    # 1. Fetch Data
    print(f"\n📥 Fetching data for {all_tickers} from {start_date.date()}...")
    fetch_start = start_date - pd.DateOffset(days=400)
    
    cache = get_cache()
    price_data = cache.get_incremental_data(all_tickers, fetch_start, end_date)
    
    if price_data.empty:
        print("❌ No data found.")
        return

    # 2. Setup Backtest Timeline
    monthly_prices = price_data.resample('ME').last()
    monthly_dates = monthly_prices.index
    monthly_dates = monthly_dates[monthly_dates >= start_date]
    
    print(f"📅 Running backtest over {len(monthly_dates)} months...")
    
    # Initialize Capitals
    strategies = ['Current', 'Forecast_1M', 'Forecast_3M', 'Forecast_6M', 'Delta']
    capitals = {s: 10000.0 for s in strategies}
    
    equity_curve = {'Date': []}
    for s in strategies:
        equity_curve[s] = []
        
    # Track defensive months
    defensive_months = 0
    
    for i in range(len(monthly_dates) - 1):
        rebal_date = monthly_dates[i]
        next_rebal_date = monthly_dates[i+1]
        
        # Data up to rebalance date
def get_universe_scores(momentum_df):
    """
    Calculates scores for all strategies for a given universe.
    Returns a dict of Series: {'Current': ..., 'Forecast_1M': ..., ...}
    """
    if momentum_df.empty:
        return {}
        
    scores = {
        'Current': momentum_df.iloc[-1],
        'Forecast_1M': pd.Series(dtype=float),
        'Forecast_3M': pd.Series(dtype=float),
        'Forecast_6M': pd.Series(dtype=float),
        'Delta': pd.Series(dtype=float)
    }
    
    current_vals = momentum_df.iloc[-1]
    
    for ticker in momentum_df.columns:
        series = momentum_df[ticker]
        f1 = calibrate_ou_and_forecast(series, months=1)
        f3 = calibrate_ou_and_forecast(series, months=3)
        f6 = calibrate_ou_and_forecast(series, months=6)
        
        scores['Forecast_1M'][ticker] = f1
        scores['Forecast_3M'][ticker] = f3
        scores['Forecast_6M'][ticker] = f6
        scores['Delta'][ticker] = f1 - current_vals[ticker]
        
    return scores

def run_backtest():
    print("🚀 Starting Backtest Comparison (Smart Protective Logic)...")
    
    # Configuration
    agg_tickers = ['SPY', 'EFA', 'EEM', 'AGG']
    prot_tickers = ['LQD', 'IEF', 'SHY']
    all_tickers = list(set(agg_tickers + prot_tickers))
    
    backtest_years = 15
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=backtest_years)
    
    # 1. Fetch Data
    print(f"\n📥 Fetching data for {all_tickers} from {start_date.date()}...")
    fetch_start = start_date - pd.DateOffset(days=400)
    
    cache = get_cache()
    price_data = cache.get_incremental_data(all_tickers, fetch_start, end_date)
    
    if price_data.empty:
        print("❌ No data found.")
        return

    # 2. Setup Backtest Timeline
    monthly_prices = price_data.resample('ME').last()
    monthly_dates = monthly_prices.index
    monthly_dates = monthly_dates[monthly_dates >= start_date]
    
    print(f"📅 Running backtest over {len(monthly_dates)} months...")
    
    # Initialize Capitals
    strategies = ['Current', 'Forecast_1M', 'Forecast_3M', 'Forecast_6M', 'Delta']
    capitals = {s: 10000.0 for s in strategies}
    
    equity_curve = {'Date': []}
    for s in strategies:
        equity_curve[s] = []
        
    # Track defensive months per strategy
    defensive_counts = {s: 0 for s in strategies}
    
    for i in range(len(monthly_dates) - 1):
        rebal_date = monthly_dates[i]
        next_rebal_date = monthly_dates[i+1]
        
        # Data up to rebalance date
        hist_data = price_data.loc[:rebal_date]
        
        # 1. Calculate Aggressive Momentum & Scores
        agg_mom_df = calculate_momentum_series(hist_data, agg_tickers)
        if agg_mom_df.empty: continue
        
        agg_scores = get_universe_scores(agg_mom_df)
        
        # 2. Calculate Protective Momentum & Scores (Lazy load if needed, but easier to just calc)
        prot_mom_df = calculate_momentum_series(hist_data, prot_tickers)
        prot_scores = get_universe_scores(prot_mom_df) if not prot_mom_df.empty else {}
        
        selections = {}
        
        for s in strategies:
            # Determine if Defensive
            # Rule: If ANY aggressive asset has negative score (using this strategy's metric), go defensive.
            # Exception: For Delta, we use Forecast_1M for safety check (absolute level), but Delta for ranking.
            
            safety_metric = s
            if s == 'Delta':
                safety_metric = 'Forecast_1M' # Use absolute level for safety
            
            # Check if any aggressive asset is negative
            is_defensive = (agg_scores[safety_metric] < 0).any()
            
            if is_defensive:
                defensive_counts[s] += 1
                # Select best from Protective
                if prot_scores:
                    # For Delta strategy in protective, we still use Delta for ranking
                    selections[s] = prot_scores[s].idxmax()
                else:
                    selections[s] = None # Should not happen
            else:
                # Select best from Aggressive
                selections[s] = agg_scores[s].idxmax()
        
        # 3. Calculate Returns
        price_start = monthly_prices.loc[rebal_date]
        price_end = monthly_prices.loc[next_rebal_date]
        
        equity_curve['Date'].append(next_rebal_date)
        
        for s in strategies:
            ticker = selections[s]
            if ticker:
                ret = (price_end[ticker] / price_start[ticker]) - 1
                capitals[s] *= (1 + ret)
            equity_curve[s].append(capitals[s])

    # 3. Analysis
    df_results = pd.DataFrame(equity_curve).set_index('Date')
    
    print("\n📊 Backtest Results (Smart Protective Logic):")
    print("=" * 60)
    print(f"{'Strategy':<15} | {'Final Balance':<15} | {'Return':<8} | {'Defensive %':<12}")
    print("-" * 60)
    
    for s in strategies:
        final_cap = capitals[s]
        total_ret = (final_cap / 10000) - 1
        def_pct = defensive_counts[s] / len(monthly_dates)
        print(f"{s:<15} | ${final_cap:,.2f}      | +{total_ret:.1%}   | {def_pct:.1%}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    for s in strategies:
        plt.plot(df_results.index, df_results[s], label=s)
        
    plt.title(f'VAA Strategy Comparison (Smart Protective Logic) - {backtest_years} Years')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtest_comparison.png')
    print("\n📈 Plot saved to 'backtest_comparison.png'")
    plt.show()

if __name__ == "__main__":
    run_backtest()
