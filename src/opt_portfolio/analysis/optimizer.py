"""
Portfolio Weight Optimizer Module

Optimizes portfolio allocation weights using Sharpe Ratio as the objective function.
Supports dynamic VAA selection where the selected asset changes monthly.

퀀트 관점:
- 최적화 목표: Sharpe Ratio 최대화
- 제약 조건: 비중 합 = 1, 최소/최대 비중 제한
- 그리드 서치 + 경사하강법 조합으로 전역 최적해 탐색
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product

from ..config import BACKTEST, ASSETS, AllocationConfig


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_weights: Dict[str, float]
    best_sharpe: float
    best_return: float
    best_volatility: float
    best_max_drawdown: float
    
    # All tested combinations for analysis
    all_results: pd.DataFrame
    
    # Optimal allocation config
    optimal_config: AllocationConfig = None
    
    def __post_init__(self):
        self.optimal_config = AllocationConfig.from_weights(
            vaa=self.best_weights.get('VAA', 0.5),
            spy=self.best_weights.get('SPY', 0.125),
            tlt=self.best_weights.get('TLT', 0.125),
            gld=self.best_weights.get('GLD', 0.125),
            bil=self.best_weights.get('BIL', 0.125)
        )
    
    def get_summary(self) -> str:
        """Get text summary of optimization result."""
        lines = [
            "=" * 60,
            "📊 PORTFOLIO WEIGHT OPTIMIZATION RESULT",
            "=" * 60,
            "",
            "🎯 Optimal Allocation:",
            f"   VAA Selected: {self.best_weights.get('VAA', 0)*100:.1f}%",
            f"   SPY: {self.best_weights.get('SPY', 0)*100:.1f}%",
            f"   TLT: {self.best_weights.get('TLT', 0)*100:.1f}%",
            f"   GLD: {self.best_weights.get('GLD', 0)*100:.1f}%",
            f"   BIL: {self.best_weights.get('BIL', 0)*100:.1f}%",
            "",
            "📈 Performance Metrics:",
            f"   Sharpe Ratio: {self.best_sharpe:.3f}",
            f"   Annual Return: {self.best_return*100:.2f}%",
            f"   Annual Volatility: {self.best_volatility*100:.2f}%",
            f"   Max Drawdown: {self.best_max_drawdown*100:.2f}%",
            "=" * 60
        ]
        return "\n".join(lines)


class PortfolioOptimizer:
    """
    Portfolio weight optimizer using grid search and Sharpe Ratio maximization.
    
    퀀트 조언:
    - Sharpe Ratio = (Return - Risk Free Rate) / Volatility
    - 그리드 서치는 전역 최적해를 놓치지 않지만 계산량 많음
    - 5% 단위 그리드는 적절한 정밀도/계산량 균형
    """
    
    def __init__(
        self,
        weight_min: float = BACKTEST.OPTIMIZATION_WEIGHT_MIN,
        weight_max: float = BACKTEST.OPTIMIZATION_WEIGHT_MAX,
        weight_step: float = BACKTEST.OPTIMIZATION_STEP,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize optimizer.
        
        Args:
            weight_min: Minimum weight for any asset
            weight_max: Maximum weight for VAA selection
            weight_step: Grid search step size
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.weight_step = weight_step
        self.risk_free_rate = risk_free_rate
    
    def generate_weight_combinations(self) -> List[Dict[str, float]]:
        """
        Generate all valid weight combinations for grid search.
        
        퀀트 조언:
        - VAA weight: 20% ~ 70% (핵심 전술적 배분)
        - Core weights: 5% ~ 30% (전략적 배분)
        - 합이 100%가 되는 조합만 유효
        
        Returns:
            List of weight dictionaries
        """
        vaa_weights = np.arange(0.20, self.weight_max + 0.01, self.weight_step)
        core_weights = np.arange(self.weight_min, 0.35, self.weight_step)
        
        combinations = []
        
        for vaa_w in vaa_weights:
            remaining = 1.0 - vaa_w
            
            # Generate core weight combinations
            for spy_w in core_weights:
                if spy_w > remaining:
                    continue
                for tlt_w in core_weights:
                    if spy_w + tlt_w > remaining:
                        continue
                    for gld_w in core_weights:
                        if spy_w + tlt_w + gld_w > remaining:
                            continue
                        
                        bil_w = remaining - spy_w - tlt_w - gld_w
                        
                        # Validate BIL weight
                        if bil_w >= self.weight_min and bil_w <= 0.35:
                            total = vaa_w + spy_w + tlt_w + gld_w + bil_w
                            if abs(total - 1.0) < 0.001:
                                combinations.append({
                                    'VAA': round(vaa_w, 3),
                                    'SPY': round(spy_w, 3),
                                    'TLT': round(tlt_w, 3),
                                    'GLD': round(gld_w, 3),
                                    'BIL': round(bil_w, 3)
                                })
        
        print(f"📊 Generated {len(combinations)} weight combinations for optimization")
        return combinations
    
    def calculate_portfolio_returns(
        self,
        vaa_returns: pd.Series,
        core_returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        Calculate portfolio returns with given weights.
        
        Args:
            vaa_returns: Monthly returns from VAA selected asset
            core_returns: DataFrame with monthly returns for SPY, TLT, GLD, BIL
            weights: Weight dictionary
            
        Returns:
            Portfolio monthly returns series
        """
        # VAA selected asset return
        portfolio_returns = vaa_returns * weights['VAA']
        
        # Add core asset returns
        for asset in ['SPY', 'TLT', 'GLD', 'BIL']:
            if asset in core_returns.columns:
                portfolio_returns += core_returns[asset] * weights[asset]
        
        return portfolio_returns
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Calculate annualized Sharpe Ratio and related metrics.
        
        퀀트 조언:
        - 연환산: 월간 수익률 × 12, 월간 변동성 × √12
        - Sharpe > 1.0 은 좋은 전략으로 간주
        - Sharpe > 2.0 은 매우 우수한 전략
        
        Args:
            returns: Monthly returns series
            
        Returns:
            Tuple of (sharpe_ratio, annual_return, annual_volatility)
        """
        if returns.empty or returns.std() == 0:
            return 0.0, 0.0, 0.0
        
        annual_return = returns.mean() * 12
        annual_volatility = returns.std() * np.sqrt(12)
        
        if annual_volatility == 0:
            return 0.0, annual_return, 0.0
        
        sharpe = (annual_return - self.risk_free_rate) / annual_volatility
        
        return sharpe, annual_return, annual_volatility
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return abs(drawdown.min())
    
    def optimize(
        self,
        vaa_returns: pd.Series,
        core_returns: pd.DataFrame,
        equity_curves: Optional[Dict[str, pd.Series]] = None
    ) -> OptimizationResult:
        """
        Find optimal portfolio weights using grid search.
        
        퀀트 조언:
        - 모든 조합에 대해 Sharpe Ratio 계산
        - 최고 Sharpe를 가진 조합 선택
        - 상위 10개 조합도 분석용으로 보관
        
        Args:
            vaa_returns: Monthly returns from VAA selected asset
            core_returns: DataFrame with SPY, TLT, GLD, BIL returns
            equity_curves: Optional dict of equity curves per asset
            
        Returns:
            OptimizationResult with best weights and metrics
        """
        print("\n🔍 Starting Portfolio Weight Optimization...")
        print("=" * 50)
        
        combinations = self.generate_weight_combinations()
        
        if not combinations:
            raise ValueError("No valid weight combinations found")
        
        results = []
        
        for i, weights in enumerate(combinations):
            if (i + 1) % 100 == 0:
                print(f"   Testing combination {i+1}/{len(combinations)}...")
            
            # Calculate portfolio returns
            port_returns = self.calculate_portfolio_returns(
                vaa_returns, core_returns, weights
            )
            
            # Calculate metrics
            sharpe, ann_return, ann_vol = self.calculate_sharpe_ratio(port_returns)
            
            # Calculate drawdown if equity curves provided
            mdd = 0.0
            if equity_curves is not None:
                # Simulate equity curve
                equity = (1 + port_returns).cumprod()
                mdd = self.calculate_max_drawdown(equity)
            
            results.append({
                **weights,
                'sharpe_ratio': sharpe,
                'annual_return': ann_return,
                'annual_volatility': ann_vol,
                'max_drawdown': mdd
            })
        
        # Convert to DataFrame and sort by Sharpe
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        # Get best result
        best = results_df.iloc[0]
        
        best_weights = {
            'VAA': best['VAA'],
            'SPY': best['SPY'],
            'TLT': best['TLT'],
            'GLD': best['GLD'],
            'BIL': best['BIL']
        }
        
        result = OptimizationResult(
            best_weights=best_weights,
            best_sharpe=best['sharpe_ratio'],
            best_return=best['annual_return'],
            best_volatility=best['annual_volatility'],
            best_max_drawdown=best['max_drawdown'],
            all_results=results_df
        )
        
        print("\n" + result.get_summary())
        
        # Print top 5 combinations
        print("\n📋 Top 5 Weight Combinations:")
        print("-" * 80)
        top5 = results_df.head(5)
        for i, row in top5.iterrows():
            print(f"   VAA:{row['VAA']*100:4.0f}% SPY:{row['SPY']*100:4.0f}% "
                  f"TLT:{row['TLT']*100:4.0f}% GLD:{row['GLD']*100:4.0f}% "
                  f"BIL:{row['BIL']*100:4.0f}% | Sharpe:{row['sharpe_ratio']:.3f}")
        
        return result


def quick_optimize(
    vaa_returns: pd.Series,
    core_returns: pd.DataFrame
) -> Dict[str, float]:
    """
    Quick optimization helper function.
    
    Args:
        vaa_returns: VAA selected asset returns
        core_returns: Core asset returns DataFrame
        
    Returns:
        Optimal weights dictionary
    """
    optimizer = PortfolioOptimizer()
    result = optimizer.optimize(vaa_returns, core_returns)
    return result.best_weights
