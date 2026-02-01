"""
Portfolio Management Core Module

This module provides the core Portfolio class for managing holdings,
calculating allocations, and tracking performance.

퀀트 관점:
- 포트폴리오 가치 계산은 실시간 가격 기준
- 리밸런싱 오차 분석으로 실제 vs 목표 괴리 추적
- 거래 비용을 고려한 최적화 필요
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..config import ALLOCATION, ASSETS


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    ticker: str
    shares: int
    average_cost: float = 0.0
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        if self.average_cost > 0:
            return (self.current_price - self.average_cost) * self.shares
        return 0.0
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.average_cost > 0:
            return ((self.current_price / self.average_cost) - 1) * 100
        return 0.0


@dataclass
class Transaction:
    """Represents a buy or sell transaction."""
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def value(self) -> float:
        """Calculate transaction value."""
        return self.shares * self.price
    
    @property
    def cost(self) -> float:
        """Calculate transaction cost (for BUY)."""
        return self.value if self.action == 'BUY' else 0.0
    
    @property
    def proceeds(self) -> float:
        """Calculate transaction proceeds (for SELL)."""
        return self.value if self.action == 'SELL' else 0.0


class Portfolio:
    """
    Portfolio management class.
    
    퀀트 조언:
    - 포트폴리오 관리의 핵심은 목표 배분 대비 현재 상태 추적
    - 정수 주식 제약으로 인한 오차는 불가피
    - 거래 비용을 고려한 최적화가 실제 수익에 중요
    """
    
    def __init__(self, holdings: Optional[Dict[str, int]] = None):
        """
        Initialize portfolio with optional holdings.
        
        Args:
            holdings: Dictionary of {ticker: shares}
        """
        self.positions: Dict[str, Position] = {}
        self.cash: float = 0.0
        self.transactions: List[Transaction] = []
        
        if holdings:
            for ticker, shares in holdings.items():
                if shares > 0:
                    self.positions[ticker] = Position(ticker=ticker, shares=shares)
    
    @classmethod
    def from_dict(cls, holdings: Dict[str, int]) -> 'Portfolio':
        """Create portfolio from holdings dictionary."""
        return cls(holdings)
    
    def update_prices(self) -> None:
        """Update current prices for all positions."""
        if not self.positions:
            return
        
        tickers = list(self.positions.keys())
        prices = self._fetch_current_prices(tickers)
        
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]
    
    def _fetch_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of {ticker: price}
        """
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
            except Exception as e:
                print(f"Warning: Could not fetch price for {ticker}: {e}")
        
        return price_data
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value including cash."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return positions_value + self.cash
    
    @property
    def positions_value(self) -> float:
        """Calculate total value of all positions."""
        return sum(p.market_value for p in self.positions.values())
    
    def get_allocation(self) -> pd.DataFrame:
        """
        Calculate current allocation percentages.
        
        Returns:
            DataFrame with allocation details
        """
        if not self.positions:
            return pd.DataFrame()
        
        self.update_prices()
        total = self.total_value
        
        data = []
        for ticker, position in self.positions.items():
            pct = (position.market_value / total * 100) if total > 0 else 0
            data.append({
                'Ticker': ticker,
                'Shares': position.shares,
                'Price': position.current_price,
                'Value': position.market_value,
                'Weight (%)': pct
            })
        
        if self.cash > 0:
            data.append({
                'Ticker': 'CASH',
                'Shares': 0,
                'Price': 1.0,
                'Value': self.cash,
                'Weight (%)': (self.cash / total * 100) if total > 0 else 0
            })
        
        df = pd.DataFrame(data)
        df.set_index('Ticker', inplace=True)
        return df
    
    def calculate_rebalance(
        self, 
        selected_etf: str, 
        additional_cash: float = 0.0,
        target_allocations: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calculate rebalancing transactions to reach target allocation.
        
        퀀트 조언:
        - 정수 주식 제약으로 완벽한 배분은 불가능
        - 우선순위: 큰 편차 먼저 교정
        - 매도 후 매수 순서로 현금 흐름 최적화
        
        Args:
            selected_etf: ETF selected by VAA strategy (gets 50%)
            additional_cash: Additional cash to invest
            target_allocations: Custom target allocations (optional)
            
        Returns:
            Dictionary with rebalancing recommendations
        """
        self.update_prices()
        
        # Get current prices including selected ETF
        all_tickers = list(self.positions.keys())
        core_tickers = list(ASSETS.CORE_TICKERS)
        
        for ticker in core_tickers + [selected_etf]:
            if ticker not in all_tickers:
                all_tickers.append(ticker)
        
        prices = self._fetch_current_prices(all_tickers)
        
        if not prices:
            return {"error": "Could not fetch price data"}
        
        # Calculate current values
        current_value = sum(
            self.positions[t].shares * prices.get(t, 0)
            for t in self.positions
            if t in prices
        )
        
        total_target_value = current_value + additional_cash
        
        # Define target allocations
        if target_allocations is None:
            target_allocations = {
                selected_etf: ALLOCATION.VAA_SELECTED_WEIGHT,
                'SPY': ALLOCATION.SPY_WEIGHT,
                'TLT': ALLOCATION.TLT_WEIGHT,
                'GLD': ALLOCATION.GLD_WEIGHT,
                'BIL': ALLOCATION.BIL_WEIGHT
            }
            
            # Remove selected_etf from core if it's there
            if selected_etf in ['SPY', 'TLT', 'GLD', 'BIL']:
                core_weight = getattr(ALLOCATION, f'{selected_etf}_WEIGHT')
                target_allocations[selected_etf] = (
                    ALLOCATION.VAA_SELECTED_WEIGHT + core_weight
                )
        
        # Calculate target values and shares
        target_values = {
            ticker: total_target_value * weight
            for ticker, weight in target_allocations.items()
        }
        
        # Calculate transactions
        transactions = {}
        total_sales = 0.0
        total_purchases = 0.0
        
        # Phase 1: Calculate sells
        for ticker in self.positions:
            if ticker not in target_allocations:
                # Sell entire position
                current_shares = self.positions[ticker].shares
                if current_shares > 0 and ticker in prices:
                    proceeds = current_shares * prices[ticker]
                    total_sales += proceeds
                    transactions[ticker] = Transaction(
                        ticker=ticker,
                        action='SELL',
                        shares=current_shares,
                        price=prices[ticker]
                    )
            else:
                # Check if we need to reduce
                current_value_ticker = self.positions[ticker].shares * prices.get(ticker, 0)
                target_value_ticker = target_values.get(ticker, 0)
                
                if current_value_ticker > target_value_ticker:
                    excess = current_value_ticker - target_value_ticker
                    shares_to_sell = int(excess / prices[ticker])
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * prices[ticker]
                        total_sales += proceeds
                        transactions[ticker] = Transaction(
                            ticker=ticker,
                            action='SELL',
                            shares=shares_to_sell,
                            price=prices[ticker]
                        )
        
        # Available cash after sells
        available_cash = additional_cash + total_sales
        
        # Phase 2: Calculate buys
        shortfalls = []
        for ticker, target_value in target_values.items():
            current_shares = self.positions.get(ticker, Position(ticker, 0)).shares
            current_value_ticker = current_shares * prices.get(ticker, 0)
            
            # Account for sells
            if ticker in transactions and transactions[ticker].action == 'SELL':
                current_value_ticker -= transactions[ticker].value
            
            shortfall = target_value - current_value_ticker
            if shortfall > 0 and ticker in prices:
                max_shares = int(shortfall / prices[ticker])
                shortfalls.append({
                    'ticker': ticker,
                    'shortfall': shortfall,
                    'max_shares': max_shares,
                    'price': prices[ticker]
                })
        
        # Sort by shortfall (largest first)
        shortfalls.sort(key=lambda x: x['shortfall'], reverse=True)
        
        # Execute purchases within cash constraints
        remaining_cash = available_cash
        for item in shortfalls:
            ticker = item['ticker']
            max_affordable = int(remaining_cash / item['price'])
            shares_to_buy = min(item['max_shares'], max_affordable)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * item['price']
                remaining_cash -= cost
                total_purchases += cost
                
                if ticker in transactions:
                    # Had a sell, update to net transaction
                    net_shares = shares_to_buy - transactions[ticker].shares
                    if net_shares > 0:
                        transactions[ticker] = Transaction(
                            ticker=ticker, action='BUY',
                            shares=net_shares, price=item['price']
                        )
                    elif net_shares < 0:
                        transactions[ticker].shares = abs(net_shares)
                    else:
                        del transactions[ticker]
                else:
                    transactions[ticker] = Transaction(
                        ticker=ticker,
                        action='BUY',
                        shares=shares_to_buy,
                        price=item['price']
                    )
        
        # Calculate final allocations
        final_portfolio = {}
        for ticker, target in target_values.items():
            current_shares = self.positions.get(ticker, Position(ticker, 0)).shares
            if ticker in transactions:
                if transactions[ticker].action == 'BUY':
                    final_portfolio[ticker] = current_shares + transactions[ticker].shares
                else:
                    final_portfolio[ticker] = current_shares - transactions[ticker].shares
            else:
                final_portfolio[ticker] = current_shares
        
        # Calculate allocation errors
        final_value = sum(final_portfolio[t] * prices.get(t, 0) for t in final_portfolio)
        allocation_errors = {}
        
        for ticker, target_weight in target_allocations.items():
            actual_value = final_portfolio.get(ticker, 0) * prices.get(ticker, 0)
            actual_weight = (actual_value / final_value * 100) if final_value > 0 else 0
            target_pct = target_weight * 100
            
            allocation_errors[ticker] = {
                'target_percentage': target_pct,
                'actual_percentage': actual_weight,
                'percentage_error': actual_weight - target_pct,
                'target_value': total_target_value * target_weight,
                'actual_value': actual_value
            }
        
        return {
            'selected_etf': selected_etf,
            'current_value': current_value,
            'additional_cash': additional_cash,
            'total_target_value': total_target_value,
            'transactions': {
                t: {
                    'action': tx.action,
                    'shares': tx.shares,
                    'price': tx.price,
                    'cost' if tx.action == 'BUY' else 'proceeds': tx.value
                }
                for t, tx in transactions.items()
            },
            'optimized_portfolio': final_portfolio,
            'allocation_errors': allocation_errors,
            'remaining_cash': remaining_cash,
            'final_portfolio_value': final_value,
            'cash_from_sales': total_sales,
            'total_available_cash': available_cash,
            'optimization_summary': {
                'total_transactions': len(transactions),
                'total_sales_proceeds': total_sales,
                'total_purchase_cost': total_purchases,
                'cash_utilization_rate': (
                    (available_cash - remaining_cash) / available_cash * 100
                ) if available_cash > 0 else 0
            }
        }
    
    def to_dict(self) -> Dict[str, int]:
        """Convert portfolio to holdings dictionary."""
        return {ticker: pos.shares for ticker, pos in self.positions.items()}
    
    def __repr__(self) -> str:
        """String representation of portfolio."""
        positions_str = ", ".join(
            f"{t}: {p.shares}" for t, p in self.positions.items()
        )
        return f"Portfolio({positions_str}, cash={self.cash:.2f})"
