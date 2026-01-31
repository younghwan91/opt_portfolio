"""
Test suite for configuration module.
"""

import pytest
from src.opt_portfolio.config import (
    AllocationConfig, 
    AssetUniverse, 
    MomentumConfig,
    OUProcessConfig,
    ASSETS, 
    ALLOCATION, 
    MOMENTUM
)


class TestAllocationConfig:
    """Tests for AllocationConfig."""
    
    def test_default_validation(self):
        """Test that default config validates."""
        config = AllocationConfig()
        assert config.validate()
    
    def test_from_weights_valid(self):
        """Test creating config from valid weights."""
        config = AllocationConfig.from_weights(
            vaa=0.4, spy=0.15, tlt=0.15, gld=0.15, bil=0.15
        )
        assert config.VAA_SELECTED_WEIGHT == 0.4
        assert config.SPY_WEIGHT == 0.15
        assert config.validate()
    
    def test_from_weights_invalid(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError):
            AllocationConfig.from_weights(
                vaa=0.5, spy=0.2, tlt=0.2, gld=0.2, bil=0.2
            )
    
    def test_target_allocations(self):
        """Test target allocations property."""
        config = AllocationConfig()
        allocations = config.target_allocations
        
        assert 'selected' in allocations
        assert 'SPY' in allocations
        assert 'TLT' in allocations
        assert 'GLD' in allocations
        assert 'BIL' in allocations
    
    def test_core_weights(self):
        """Test core weights property."""
        config = AllocationConfig()
        core = config.core_weights
        
        assert len(core) == 4
        assert all(k in core for k in ['SPY', 'TLT', 'GLD', 'BIL'])


class TestAssetUniverse:
    """Tests for AssetUniverse."""
    
    def test_aggressive_tickers(self):
        """Test aggressive tickers are defined."""
        assert len(ASSETS.AGGRESSIVE_TICKERS) == 4
        assert 'SPY' in ASSETS.AGGRESSIVE_TICKERS
        assert 'EFA' in ASSETS.AGGRESSIVE_TICKERS
        assert 'EEM' in ASSETS.AGGRESSIVE_TICKERS
        assert 'AGG' in ASSETS.AGGRESSIVE_TICKERS
    
    def test_protective_tickers(self):
        """Test protective tickers are defined."""
        assert len(ASSETS.PROTECTIVE_TICKERS) == 3
        assert 'LQD' in ASSETS.PROTECTIVE_TICKERS
        assert 'IEF' in ASSETS.PROTECTIVE_TICKERS
        assert 'SHY' in ASSETS.PROTECTIVE_TICKERS
    
    def test_core_tickers(self):
        """Test core tickers are defined."""
        assert len(ASSETS.CORE_TICKERS) == 4
        assert 'SPY' in ASSETS.CORE_TICKERS
        assert 'TLT' in ASSETS.CORE_TICKERS
        assert 'GLD' in ASSETS.CORE_TICKERS
        assert 'BIL' in ASSETS.CORE_TICKERS


class TestMomentumConfig:
    """Tests for MomentumConfig."""
    
    def test_trading_days(self):
        """Test trading days constants."""
        assert MOMENTUM.TRADING_DAYS_PER_MONTH == 21
        assert MOMENTUM.TRADING_DAYS_PER_YEAR == 252
    
    def test_period_days(self):
        """Test period days constants."""
        assert MOMENTUM.PERIOD_1M_DAYS == 21
        assert MOMENTUM.PERIOD_3M_DAYS == 63
        assert MOMENTUM.PERIOD_6M_DAYS == 126
        assert MOMENTUM.PERIOD_12M_DAYS == 252
    
    def test_weights(self):
        """Test momentum weights (VAA formula)."""
        assert MOMENTUM.WEIGHT_1M == 12
        assert MOMENTUM.WEIGHT_3M == 4
        assert MOMENTUM.WEIGHT_6M == 2
        assert MOMENTUM.WEIGHT_12M == 1
    
    def test_weight_sum(self):
        """Test total weight sum is 19."""
        total = (MOMENTUM.WEIGHT_1M + MOMENTUM.WEIGHT_3M + 
                 MOMENTUM.WEIGHT_6M + MOMENTUM.WEIGHT_12M)
        assert total == 19


class TestOUProcessConfig:
    """Tests for OUProcessConfig."""
    
    def test_theta_bounds(self):
        """Test theta bounds are reasonable."""
        from src.opt_portfolio.config import OU_PROCESS
        assert OU_PROCESS.THETA_MIN > 0
        assert OU_PROCESS.THETA_MAX > OU_PROCESS.THETA_MIN
        assert OU_PROCESS.THETA_MAX <= 1.0
    
    def test_slope_bounds(self):
        """Test slope bounds are valid."""
        from src.opt_portfolio.config import OU_PROCESS
        assert 0 < OU_PROCESS.SLOPE_MIN < 1
        assert 0 < OU_PROCESS.SLOPE_MAX < 1
        assert OU_PROCESS.SLOPE_MAX > OU_PROCESS.SLOPE_MIN
    
    def test_simulation_defaults(self):
        """Test simulation default values."""
        from src.opt_portfolio.config import OU_PROCESS
        assert OU_PROCESS.DEFAULT_SIMULATIONS >= 100
        assert OU_PROCESS.DEFAULT_FORECAST_MONTHS >= 1
