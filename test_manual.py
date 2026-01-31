"""
Manual Test Script for Portfolio Optimization

This script tests core functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test configuration module."""
    print("=" * 60)
    print("TEST 1: Configuration Module")
    print("=" * 60)
    
    from src.opt_portfolio.config import AllocationConfig, ASSETS, MOMENTUM
    
    # Test default config
    config = AllocationConfig()
    print(f"✓ Default config created")
    print(f"  VAA Weight: {config.VAA_SELECTED_WEIGHT * 100:.1f}%")
    print(f"  SPY Weight: {config.SPY_WEIGHT * 100:.1f}%")
    print(f"  TLT Weight: {config.TLT_WEIGHT * 100:.1f}%")
    print(f"  GLD Weight: {config.GLD_WEIGHT * 100:.1f}%")
    print(f"  BIL Weight: {config.BIL_WEIGHT * 100:.1f}%")
    
    assert config.validate(), "Default config should be valid"
    print(f"✓ Config validation passed")
    
    # Test custom config
    custom_config = AllocationConfig.from_weights(
        vaa=0.4, spy=0.15, tlt=0.15, gld=0.15, bil=0.15
    )
    print(f"✓ Custom config created (VAA 40%, others 15%)")
    assert custom_config.validate(), "Custom config should be valid"
    
    # Test asset universe
    print(f"\n✓ Asset Universe:")
    print(f"  Aggressive: {', '.join(ASSETS.AGGRESSIVE_TICKERS)}")
    print(f"  Protective: {', '.join(ASSETS.PROTECTIVE_TICKERS)}")
    print(f"  Core: {', '.join(ASSETS.CORE_TICKERS)}")
    
    # Test momentum config
    print(f"\n✓ Momentum Weights: {MOMENTUM.WEIGHT_1M}, {MOMENTUM.WEIGHT_3M}, {MOMENTUM.WEIGHT_6M}, {MOMENTUM.WEIGHT_12M}")
    total_weight = MOMENTUM.WEIGHT_1M + MOMENTUM.WEIGHT_3M + MOMENTUM.WEIGHT_6M + MOMENTUM.WEIGHT_12M
    assert total_weight == 19, f"Total weight should be 19, got {total_weight}"
    
    print("\n✅ Configuration tests PASSED\n")
    return True


def test_optimizer():
    """Test portfolio optimizer."""
    print("=" * 60)
    print("TEST 2: Portfolio Optimizer")
    print("=" * 60)
    
    from src.opt_portfolio.analysis.optimizer import PortfolioOptimizer
    import pandas as pd
    import numpy as np
    
    # Create optimizer
    optimizer = PortfolioOptimizer(
        weight_min=0.05,
        weight_max=0.70,
        weight_step=0.10  # Larger step for faster testing
    )
    print(f"✓ Optimizer created")
    
    # Generate weight combinations
    combinations = optimizer.generate_weight_combinations()
    print(f"✓ Generated {len(combinations)} weight combinations")
    assert len(combinations) > 0, "Should generate at least one combination"
    
    # Verify first combination sums to 1.0
    first = combinations[0]
    total = sum(first.values())
    print(f"✓ First combination: {first}")
    print(f"  Sum: {total:.3f}")
    assert abs(total - 1.0) < 0.001, f"Weights should sum to 1.0, got {total}"
    
    # Create synthetic returns for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    
    # VAA returns (slightly positive)
    vaa_returns = pd.Series(
        np.random.normal(0.01, 0.03, 60),
        index=dates,
        name='VAA'
    )
    
    # Core returns
    core_returns = pd.DataFrame({
        'SPY': np.random.normal(0.01, 0.04, 60),
        'TLT': np.random.normal(0.005, 0.02, 60),
        'GLD': np.random.normal(0.005, 0.03, 60),
        'BIL': np.random.normal(0.001, 0.005, 60)
    }, index=dates)
    
    print(f"\n✓ Created synthetic returns:")
    print(f"  VAA mean: {vaa_returns.mean():.4f}")
    print(f"  SPY mean: {core_returns['SPY'].mean():.4f}")
    
    # Test portfolio return calculation
    test_weights = {'VAA': 0.5, 'SPY': 0.125, 'TLT': 0.125, 'GLD': 0.125, 'BIL': 0.125}
    port_returns = optimizer.calculate_portfolio_returns(vaa_returns, core_returns, test_weights)
    print(f"\n✓ Portfolio returns calculated:")
    print(f"  Mean return: {port_returns.mean():.4f}")
    print(f"  Std dev: {port_returns.std():.4f}")
    
    # Test Sharpe ratio calculation
    sharpe, ann_return, ann_vol = optimizer.calculate_sharpe_ratio(port_returns)
    print(f"\n✓ Sharpe ratio calculated:")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Annual Return: {ann_return:.2%}")
    print(f"  Annual Volatility: {ann_vol:.2%}")
    
    print("\n✅ Optimizer tests PASSED\n")
    return True


def test_allocation_config():
    """Test allocation config edge cases."""
    print("=" * 60)
    print("TEST 3: AllocationConfig Edge Cases")
    print("=" * 60)
    
    from src.opt_portfolio.config import AllocationConfig
    
    # Test invalid sum
    try:
        AllocationConfig.from_weights(
            vaa=0.5, spy=0.2, tlt=0.2, gld=0.2, bil=0.2
        )
        print("❌ Should have raised ValueError for invalid sum")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid weights: {e}")
    
    # Test boundary values
    config = AllocationConfig.from_weights(
        vaa=0.20, spy=0.20, tlt=0.20, gld=0.20, bil=0.20
    )
    print(f"✓ Equal weights (20% each) accepted")
    assert config.validate()
    
    # Test core_weights property
    core = config.core_weights
    print(f"✓ Core weights: {core}")
    assert len(core) == 4
    assert all(k in core for k in ['SPY', 'TLT', 'GLD', 'BIL'])
    
    # Test target_allocations property
    targets = config.target_allocations
    print(f"✓ Target allocations: {targets}")
    assert 'selected' in targets
    
    print("\n✅ Edge case tests PASSED\n")
    return True


def test_backtest_result():
    """Test BacktestResult dataclass."""
    print("=" * 60)
    print("TEST 4: BacktestResult")
    print("=" * 60)
    
    from src.opt_portfolio.analysis.backtest import BacktestResult
    import pandas as pd
    import numpy as np
    
    # Create test data
    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    equity = pd.Series(
        np.cumprod(1 + np.random.normal(0.01, 0.03, 60)) * 10000,
        index=dates
    )
    returns = pd.Series(
        np.random.normal(0.01, 0.03, 60),
        index=dates
    )
    
    result = BacktestResult(
        strategy_name="Test Strategy",
        initial_capital=10000.0,
        final_capital=equity.iloc[-1],
        equity_curve=equity,
        returns=returns,
        transactions=[],
        vaa_selections=['SPY'] * 30 + ['AGG'] * 30
    )
    
    print(f"✓ BacktestResult created")
    print(f"  Strategy: {result.strategy_name}")
    print(f"  Initial: ${result.initial_capital:,.0f}")
    print(f"  Final: ${result.final_capital:,.0f}")
    
    # Calculate metrics
    result.calculate_metrics(years=5)
    print(f"\n✓ Metrics calculated:")
    print(f"  CAGR: {result.cagr:.2%}")
    print(f"  Sharpe: {result.sharpe_ratio:.3f}")
    print(f"  Max DD: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    
    # Test selection summary
    summary = result.get_selection_summary()
    print(f"\n✓ Selection summary:")
    print(summary)
    
    print("\n✅ BacktestResult tests PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("🧪 PORTFOLIO OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        ("Configuration", test_config),
        ("Portfolio Optimizer", test_optimizer),
        ("Allocation Config Edge Cases", test_allocation_config),
        ("Backtest Result", test_backtest_result),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}\n")
    
    print("=" * 60)
    print(f"TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
