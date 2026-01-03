"""
Smart caching system for financial market data using DuckDB.

This module provides efficient storage and retrieval of historical price data
with automatic incremental updates to minimize redundant API calls.
"""

import duckdb
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta


class DataCache:
    """
    DuckDB-based cache manager for historical market data.
    
    Features:
    - Fast columnar storage optimized for time series data
    - Automatic incremental updates (only fetch missing data)
    - Metadata tracking for cache freshness
    - Built-in cache management utilities
    """
    
    def __init__(self, db_path=".cache/market_data.duckdb"):
        """Initialize cache with DuckDB database."""
        cache_dir = Path(db_path).parent
        cache_dir.mkdir(exist_ok=True)
        
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self._init_database()
    
    def _init_database(self):
        """Create database schema for price data and metadata."""
        # Table for historical price data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                close DOUBLE NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Table for tracking cache metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                ticker VARCHAR PRIMARY KEY,
                earliest_date DATE,
                latest_date DATE,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                record_count INTEGER
            )
        """)
        
        # Index for faster date-based queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_date 
            ON price_data(date, ticker)
        """)
    
    def get_data(self, tickers, start_date, end_date):
        """
        Retrieve cached price data for given tickers and date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
        
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Build query with proper parameterization
        placeholders = ', '.join(['?' for _ in tickers])
        query = f"""
            SELECT date, ticker, close
            FROM price_data
            WHERE ticker IN ({placeholders})
            AND date BETWEEN ? AND ?
            ORDER BY date, ticker
        """
        
        try:
            result = self.conn.execute(
                query, 
                [*tickers, str(start_date), str(end_date)]
            ).df()
            
            if result.empty:
                return pd.DataFrame()
            
            # Pivot to wide format (dates x tickers)
            df = result.pivot(index='date', columns='ticker', values='close')
            df.index = pd.to_datetime(df.index)
            
            return df
        except Exception as e:
            print(f"⚠️ Error reading from cache: {e}")
            return pd.DataFrame()
    
    def save_data(self, df, tickers):
        """
        Save price data to cache.
        
        Args:
            df: DataFrame with dates as index and tickers as columns
            tickers: List of ticker symbols (used for validation)
        """
        if df.empty:
            return
        
        try:
            # Convert to long format for database storage
            df_long = df.reset_index()
            df_long.columns = ['date'] + list(df.columns)
            df_long = df_long.melt(
                id_vars=['date'], 
                var_name='ticker', 
                value_name='close'
            )
            
            # Remove any NaN values
            df_long = df_long.dropna(subset=['close'])
            
            # Convert date to proper format
            df_long['date'] = pd.to_datetime(df_long['date']).dt.date
            
            # Insert or update data
            self.conn.execute("""
                INSERT OR REPLACE INTO price_data (ticker, date, close, last_updated)
                SELECT ticker, date, close, CURRENT_TIMESTAMP
                FROM df_long
            """)
            
            # Update metadata for each ticker
            for ticker in df_long['ticker'].unique():
                self.conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata 
                    (ticker, earliest_date, latest_date, last_updated, record_count)
                    SELECT 
                        ?,
                        MIN(date),
                        MAX(date),
                        CURRENT_TIMESTAMP,
                        COUNT(*)
                    FROM price_data
                    WHERE ticker = ?
                """, [ticker, ticker])
            
            self.conn.commit()
            print(f"💾 Cached {len(df_long)} price records")
            
        except Exception as e:
            print(f"⚠️ Error saving to cache: {e}")
    
    def get_missing_date_ranges(self, ticker, start_date, end_date):
        """
        Determine what date ranges are missing from cache for a ticker.
        
        Returns:
            List of (start, end) tuples for missing ranges
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Check metadata
        meta = self.conn.execute("""
            SELECT earliest_date, latest_date
            FROM cache_metadata
            WHERE ticker = ?
        """, [ticker]).df()
        
        if meta.empty:
            # No data cached at all
            return [(start_date, end_date)]
        
        cached_start = pd.to_datetime(meta['earliest_date'].iloc[0])
        cached_end = pd.to_datetime(meta['latest_date'].iloc[0])
        
        missing_ranges = []
        
        # Need data before cached range
        if start_date < cached_start:
            missing_ranges.append((start_date, cached_start - timedelta(days=1)))
        
        # Need data after cached range
        if end_date > cached_end:
            missing_ranges.append((cached_end + timedelta(days=1), end_date))
        
        return missing_ranges
    
    def get_incremental_data(self, tickers, start_date, end_date):
        """
        Smart data fetching with incremental updates.
        Only downloads data that's not already cached.
        
        Args:
            tickers: List of ticker symbols
            start_date: Desired start date
            end_date: Desired end date
        
        Returns:
            Complete DataFrame with all requested data
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Check what data we already have
        cached_data = self.get_data(tickers, start_date, end_date)
        
        # Determine what's missing for each ticker
        tickers_to_fetch = []
        for ticker in tickers:
            missing_ranges = self.get_missing_date_ranges(ticker, start_date, end_date)
            if missing_ranges:
                tickers_to_fetch.append(ticker)
        
        if not tickers_to_fetch:
            print(f"✅ Using cached data for {', '.join(tickers)}")
            return cached_data
        
        # Fetch missing data
        print(f"📥 Fetching data for: {', '.join(tickers_to_fetch)}")
        try:
            new_data = yf.download(
                tickers_to_fetch,
                start=start_date,
                end=end_date + timedelta(days=1),
                auto_adjust=True,
                progress=False
            )
            
            if 'Close' in new_data.columns:
                new_data = new_data['Close']
            
            if isinstance(new_data, pd.Series):
                new_data = new_data.to_frame(name=tickers_to_fetch[0])
            
            # Save new data to cache
            self.save_data(new_data, tickers_to_fetch)
            
            # Retrieve complete dataset from cache
            complete_data = self.get_data(tickers, start_date, end_date)
            return complete_data
            
        except Exception as e:
            print(f"⚠️ Error fetching data: {e}")
            return cached_data if not cached_data.empty else pd.DataFrame()
    
    def get_cache_stats(self):
        """Get statistics about cached data."""
        try:
            stats = self.conn.execute("""
                SELECT 
                    ticker,
                    earliest_date,
                    latest_date,
                    record_count,
                    last_updated
                FROM cache_metadata
                ORDER BY ticker
            """).df()
            
            return stats
        except Exception as e:
            print(f"⚠️ Error getting cache stats: {e}")
            return pd.DataFrame()
    
    def clear_cache(self, older_than_days=None, tickers=None):
        """
        Clear cached data.
        
        Args:
            older_than_days: Only clear data older than this (optional)
            tickers: Only clear specific tickers (optional)
        """
        try:
            if tickers:
                # Clear specific tickers
                placeholders = ', '.join(['?' for _ in tickers])
                self.conn.execute(
                    f"DELETE FROM price_data WHERE ticker IN ({placeholders})",
                    tickers
                )
                self.conn.execute(
                    f"DELETE FROM cache_metadata WHERE ticker IN ({placeholders})",
                    tickers
                )
                print(f"🗑️ Cleared cache for: {', '.join(tickers)}")
            elif older_than_days:
                # Clear old data
                cutoff = datetime.now() - timedelta(days=older_than_days)
                result = self.conn.execute(
                    "DELETE FROM price_data WHERE last_updated < ?",
                    [cutoff]
                ).fetchall()
                self.conn.execute(
                    "DELETE FROM cache_metadata WHERE last_updated < ?",
                    [cutoff]
                )
                print(f"🗑️ Cleared cache older than {older_than_days} days")
            else:
                # Clear everything
                self.conn.execute("DELETE FROM price_data")
                self.conn.execute("DELETE FROM cache_metadata")
                print("🗑️ Cleared all cache data")
            
            self.conn.commit()
            
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")
    
    def optimize(self):
        """Optimize database for better performance."""
        try:
            self.conn.execute("VACUUM")
            self.conn.execute("ANALYZE")
            print("✨ Cache database optimized")
        except Exception as e:
            print(f"⚠️ Error optimizing database: {e}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# Global cache instance
_cache = None

def get_cache():
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache


def clear_global_cache():
    """Clear and reset the global cache instance."""
    global _cache
    if _cache is not None:
        _cache.close()
        _cache = None
