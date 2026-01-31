"""
Utility helper functions for the portfolio management system.
"""

from typing import Union, Optional
import pandas as pd


def format_currency(value: Union[float, int], symbol: str = "$") -> str:
    """Format a number as currency."""
    return f"{symbol}{value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: Union[float, int], decimals: int = 2) -> str:
    """Format a number with specified decimals."""
    return f"{value:,.{decimals}f}"


def validate_ticker(ticker: str) -> bool:
    """
    Validate a ticker symbol.
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic validation: alphanumeric, 1-5 characters
    ticker = ticker.strip().upper()
    return ticker.isalpha() and 1 <= len(ticker) <= 5


def calculate_trading_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """
    Calculate approximate trading days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Approximate number of trading days
    """
    total_days = (end_date - start_date).days
    # Rough estimate: 252 trading days per 365 calendar days
    return int(total_days * 252 / 365)


def annualize_return(total_return: float, years: float) -> float:
    """
    Convert total return to annualized return.
    
    Args:
        total_return: Total return as decimal (e.g., 0.5 for 50%)
        years: Number of years
        
    Returns:
        Annualized return as decimal
    """
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def calculate_portfolio_value(holdings: dict, prices: dict) -> float:
    """
    Calculate total portfolio value.
    
    Args:
        holdings: Dictionary of {ticker: shares}
        prices: Dictionary of {ticker: price}
        
    Returns:
        Total portfolio value
    """
    total = 0.0
    for ticker, shares in holdings.items():
        if ticker in prices and shares > 0:
            total += shares * prices[ticker]
    return total


def get_color_for_value(
    value: float, 
    thresholds: tuple = (0, 0), 
    colors: tuple = ("green", "gray", "red")
) -> str:
    """
    Get color based on value thresholds.
    
    Args:
        value: Value to evaluate
        thresholds: (positive_threshold, negative_threshold)
        colors: (positive_color, neutral_color, negative_color)
        
    Returns:
        Color string
    """
    if value > thresholds[0]:
        return colors[0]
    elif value < thresholds[1]:
        return colors[2]
    return colors[1]
