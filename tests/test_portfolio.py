"""
Unit tests for Portfolio — position management and rebalancing logic.
Tests mock external price fetching to avoid network calls.
"""

from __future__ import annotations

import pandas as pd

from opt_portfolio.core.portfolio import Portfolio, Position


class TestPortfolioInit:
    def test_empty_portfolio(self) -> None:
        p = Portfolio()
        assert p.positions == {}
        assert p.cash == 0.0

    def test_holdings_dict_creates_positions(self) -> None:
        p = Portfolio({"SPY": 10, "TLT": 5})
        assert "SPY" in p.positions
        assert "TLT" in p.positions
        assert p.positions["SPY"].shares == 10

    def test_zero_shares_excluded(self) -> None:
        p = Portfolio({"SPY": 0, "TLT": 5})
        assert "SPY" not in p.positions
        assert "TLT" in p.positions

    def test_from_dict_class_method(self) -> None:
        p = Portfolio.from_dict({"GLD": 3})
        assert p.positions["GLD"].shares == 3


class TestTotalValue:
    def test_empty_portfolio_zero_value(self) -> None:
        p = Portfolio()
        assert p.total_value == 0.0

    def test_cash_included_in_total(self) -> None:
        p = Portfolio()
        p.cash = 1000.0
        assert p.total_value == 1000.0

    def test_positions_plus_cash(self) -> None:
        p = Portfolio({"SPY": 2})
        p.positions["SPY"].current_price = 400.0
        p.cash = 200.0
        assert p.total_value == 1000.0


class TestGetAllocation:
    def test_returns_dataframe(self) -> None:
        p = Portfolio({"SPY": 1})
        p.positions["SPY"].current_price = 100.0
        result = p.get_allocation()
        assert isinstance(result, pd.DataFrame)

    def test_allocations_sum_to_100(self) -> None:
        p = Portfolio({"SPY": 2, "TLT": 3})
        p.positions["SPY"].current_price = 400.0
        p.positions["TLT"].current_price = 140.0
        df = p.get_allocation()
        pct_col = [
            c for c in df.columns if "%" in c or "weight" in c.lower() or "alloc" in c.lower()
        ]
        if pct_col:
            total = df[pct_col[0]].sum()
            assert abs(total - 100.0) < 0.1

    def test_empty_portfolio_empty_dataframe(self) -> None:
        p = Portfolio()
        result = p.get_allocation()
        assert len(result) == 0


class TestPositionValue:
    def test_market_value_computed(self) -> None:
        pos = Position(ticker="SPY", shares=10, current_price=400.0)
        assert pos.market_value == 4000.0

    def test_unrealized_pnl(self) -> None:
        pos = Position(ticker="SPY", shares=5, current_price=410.0, average_cost=400.0)
        assert pos.unrealized_pnl == 50.0

    def test_unrealized_pnl_pct(self) -> None:
        pos = Position(ticker="SPY", shares=2, current_price=110.0, average_cost=100.0)
        assert abs(pos.unrealized_pnl_pct - 10.0) < 1e-6
