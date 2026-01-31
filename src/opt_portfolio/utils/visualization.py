"""
Visualization utilities for portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import matplotlib.pyplot as plt


def create_equity_chart(
    equity_curves: Dict[str, pd.Series],
    title: str = "Equity Curves",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Create equity curve comparison chart.
    
    Args:
        equity_curves: Dictionary of {name: equity_series}
        title: Chart title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=name, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_allocation_pie(
    allocations: Dict[str, float],
    title: str = "Portfolio Allocation",
    figsize: tuple = (8, 8)
) -> plt.Figure:
    """
    Create allocation pie chart.
    
    Args:
        allocations: Dictionary of {asset: weight}
        title: Chart title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(allocations.keys())
    sizes = list(allocations.values())
    
    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85
    )
    
    # Style
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_drawdown_chart(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Create drawdown chart.
    
    Args:
        equity_curve: Equity curve series
        title: Chart title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_momentum_chart(
    momentum_df: pd.DataFrame,
    title: str = "Momentum History",
    forecast_df: Optional[pd.DataFrame] = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Create momentum history chart with optional forecast.
    
    Args:
        momentum_df: Historical momentum DataFrame
        title: Chart title
        forecast_df: Optional forecast DataFrame
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(range(len(momentum_df.columns)))
    color_map = {col: colors[i] for i, col in enumerate(momentum_df.columns)}
    
    for column in momentum_df.columns:
        color = color_map[column]
        ax.plot(momentum_df.index, momentum_df[column], label=column, 
                color=color, linewidth=2)
        
        if forecast_df is not None and column in forecast_df.columns:
            ax.plot(forecast_df.index, forecast_df[column], 
                   linestyle='--', color=color, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    if not momentum_df.empty:
        ax.axvline(x=momentum_df.index[-1], color='gray', 
                  linestyle=':', alpha=0.5, label='Today')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Momentum Score')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_returns_histogram(
    returns: pd.Series,
    title: str = "Returns Distribution",
    bins: int = 50,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create returns distribution histogram.
    
    Args:
        returns: Returns series
        title: Chart title
        bins: Number of bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(returns * 100, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=returns.mean() * 100, color='red', linestyle='-', 
               label=f'Mean: {returns.mean()*100:.2f}%')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_performance_summary(
    results: Dict,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Create performance summary chart with multiple metrics.
    
    Args:
        results: Dictionary of backtest results
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    strategies = list(results.keys())
    
    # CAGR
    ax1 = axes[0, 0]
    cagrs = [results[s].cagr * 100 for s in strategies]
    ax1.bar(strategies, cagrs, color='green', alpha=0.7)
    ax1.set_title('CAGR (%)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Sharpe Ratio
    ax2 = axes[0, 1]
    sharpes = [results[s].sharpe_ratio for s in strategies]
    ax2.bar(strategies, sharpes, color='blue', alpha=0.7)
    ax2.set_title('Sharpe Ratio', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Max Drawdown
    ax3 = axes[1, 0]
    mdds = [results[s].max_drawdown * 100 for s in strategies]
    ax3.bar(strategies, mdds, color='red', alpha=0.7)
    ax3.set_title('Max Drawdown (%)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Win Rate
    ax4 = axes[1, 1]
    wins = [results[s].win_rate * 100 for s in strategies]
    ax4.bar(strategies, wins, color='purple', alpha=0.7)
    ax4.set_title('Win Rate (%)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
