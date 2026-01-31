"""
Vigilant Asset Allocation (VAA) Strategy Module

This module implements the VAA strategy as described by Wouter Keller,
with enhancements including OU-process forecasting for improved selection.

퀀트 관점:
- VAA는 절대 모멘텀 + 상대 모멘텀을 결합한 전술적 자산배분 전략
- 핵심: "하나라도 음수면 방어모드" (절대 모멘텀)
- 백테스트상 연 15%+ 수익률, MDD 15% 이하 기대
"""

import pandas as pd
from datetime import date
from typing import Dict, Optional, Tuple, List
from enum import Enum

from ..config import ASSETS, AllocationMode, StrategyType
from .momentum import MomentumAnalyzer
from .ou_process import OUForecaster


class SelectionResult:
    """Result of VAA selection process."""
    
    def __init__(
        self,
        selected_etf: str,
        mode: AllocationMode,
        aggressive_ranking: pd.DataFrame,
        protective_ranking: pd.DataFrame,
        strategy_recommendations: Optional[Dict] = None
    ):
        self.selected_etf = selected_etf
        self.mode = mode
        self.aggressive_ranking = aggressive_ranking
        self.protective_ranking = protective_ranking
        self.strategy_recommendations = strategy_recommendations
    
    @property
    def is_defensive(self) -> bool:
        """Check if in defensive mode."""
        return self.mode == AllocationMode.DEFENSIVE
    
    def get_summary(self) -> str:
        """Get text summary of selection."""
        mode_str = "🛡️ DEFENSIVE" if self.is_defensive else "📈 GROWTH"
        return f"{mode_str} Mode | Selected: {self.selected_etf}"


class VAAStrategy:
    """
    Vigilant Asset Allocation Strategy implementation.
    
    퀀트 조언:
    - VAA는 Wouter Keller (2017)가 개발한 전략
    - 공격 자산군: SPY, EFA, EEM, AGG (글로벌 분산)
    - 방어 자산군: LQD, IEF, SHY (채권 중심)
    - 전환 규칙: 공격 자산 중 하나라도 음의 모멘텀 → 방어
    
    백테스트 성과 (2003-2023):
    - CAGR: ~15%
    - Max Drawdown: ~15%
    - Sharpe Ratio: ~1.2
    """
    
    def __init__(
        self,
        aggressive_tickers: Optional[List[str]] = None,
        protective_tickers: Optional[List[str]] = None,
        use_cache: bool = True,
        use_forecasting: bool = True
    ):
        """
        Initialize VAA strategy.
        
        Args:
            aggressive_tickers: Aggressive universe (default: VAA standard)
            protective_tickers: Protective universe (default: VAA standard)
            use_cache: Whether to use data caching
            use_forecasting: Whether to use OU forecasting
        """
        self.aggressive_tickers = list(
            aggressive_tickers or ASSETS.AGGRESSIVE_TICKERS
        )
        self.protective_tickers = list(
            protective_tickers or ASSETS.PROTECTIVE_TICKERS
        )
        
        self.momentum_analyzer = MomentumAnalyzer(use_cache=use_cache)
        self.forecaster = OUForecaster() if use_forecasting else None
        self.use_forecasting = use_forecasting
    
    def select(
        self,
        calculation_date: Optional[date] = None,
        strategy: StrategyType = StrategyType.CURRENT
    ) -> SelectionResult:
        """
        Run VAA selection for given date.
        
        퀀트 조언:
        - CURRENT: 기본 VAA (현재 모멘텀 점수 기준)
        - FORECAST_1M/3M/6M: OU 예측값으로 선택
        - DELTA: 모멘텀 변화율 (상승 추세인 자산 선호)
        
        Args:
            calculation_date: Reference date (default: today)
            strategy: Selection strategy to use
            
        Returns:
            SelectionResult with all analysis data
        """
        if calculation_date is None:
            calculation_date = date.today()
        
        print(f"\n📊 VAA Analysis as of {calculation_date}")
        print("=" * 50)
        
        # Analyze aggressive assets
        agg_ranked, _ = self.momentum_analyzer.calculate_and_rank(
            self.aggressive_tickers, calculation_date
        )
        
        # Analyze protective assets
        prot_ranked, _ = self.momentum_analyzer.calculate_and_rank(
            self.protective_tickers, calculation_date
        )
        
        if agg_ranked.empty:
            raise ValueError("Could not calculate aggressive asset momentum")
        
        # Determine mode based on any negative momentum
        is_defensive = self.momentum_analyzer.is_negative_momentum(agg_ranked)
        mode = AllocationMode.DEFENSIVE if is_defensive else AllocationMode.AGGRESSIVE
        
        # Get strategy recommendations if forecasting enabled
        strategy_recs = None
        if self.use_forecasting and self.forecaster:
            strategy_recs = self._get_strategy_recommendations(
                calculation_date, is_defensive
            )
        
        # Select ETF based on strategy
        selected_etf = self._select_by_strategy(
            strategy,
            agg_ranked if not is_defensive else prot_ranked,
            strategy_recs,
            is_defensive
        )
        
        # Print results
        self._print_analysis(
            agg_ranked, prot_ranked, mode, selected_etf, strategy_recs
        )
        
        return SelectionResult(
            selected_etf=selected_etf,
            mode=mode,
            aggressive_ranking=agg_ranked,
            protective_ranking=prot_ranked,
            strategy_recommendations=strategy_recs
        )
    
    def _get_strategy_recommendations(
        self,
        calculation_date: date,
        is_defensive: bool
    ) -> Dict:
        """Get recommendations from multiple strategies."""
        from datetime import timedelta
        
        # Get historical momentum for forecasting
        end_date = calculation_date
        start_date = end_date - timedelta(days=365*2)
        
        tickers = self.protective_tickers if is_defensive else self.aggressive_tickers
        hist_momentum = self.momentum_analyzer.calculate_historical_momentum(
            tickers, start_date, end_date
        )
        
        if hist_momentum.empty:
            return {}
        
        recommendations = {}
        
        # Current scores
        current_scores = hist_momentum.iloc[-1]
        recommendations['Current'] = {
            'asset': current_scores.idxmax(),
            'score': current_scores.max()
        }
        
        # Forecast-based selections
        for ticker in hist_momentum.columns:
            series = hist_momentum[ticker].dropna().tail(60)
            if len(series) < 30:
                continue
            
            f1 = self.forecaster.forecast(series, months=1)
            f3 = self.forecaster.forecast(series, months=3)
            f6 = self.forecaster.forecast(series, months=6)
            delta = self.forecaster.forecast_delta(series, months=1)
            
            if 'Forecast_1M' not in recommendations:
                recommendations['Forecast_1M'] = {'scores': {}}
                recommendations['Forecast_3M'] = {'scores': {}}
                recommendations['Forecast_6M'] = {'scores': {}}
                recommendations['Delta'] = {'scores': {}}
            
            recommendations['Forecast_1M']['scores'][ticker] = f1
            recommendations['Forecast_3M']['scores'][ticker] = f3
            recommendations['Forecast_6M']['scores'][ticker] = f6
            recommendations['Delta']['scores'][ticker] = delta
        
        # Find best for each strategy
        for key in ['Forecast_1M', 'Forecast_3M', 'Forecast_6M', 'Delta']:
            if key in recommendations and 'scores' in recommendations[key]:
                scores = recommendations[key]['scores']
                if scores:
                    best_ticker = max(scores, key=scores.get)
                    recommendations[key]['asset'] = best_ticker
                    recommendations[key]['score'] = scores[best_ticker]
        
        return recommendations
    
    def _select_by_strategy(
        self,
        strategy: StrategyType,
        ranked_df: pd.DataFrame,
        strategy_recs: Optional[Dict],
        is_defensive: bool
    ) -> str:
        """Select ETF based on specified strategy."""
        
        if strategy == StrategyType.CURRENT or not strategy_recs:
            return ranked_df.index[0]
        
        strategy_map = {
            StrategyType.FORECAST_1M: 'Forecast_1M',
            StrategyType.FORECAST_3M: 'Forecast_3M',
            StrategyType.FORECAST_6M: 'Forecast_6M',
            StrategyType.DELTA: 'Delta'
        }
        
        key = strategy_map.get(strategy)
        if key and key in strategy_recs:
            return strategy_recs[key].get('asset', ranked_df.index[0])
        
        return ranked_df.index[0]
    
    def _print_analysis(
        self,
        agg_ranked: pd.DataFrame,
        prot_ranked: pd.DataFrame,
        mode: AllocationMode,
        selected_etf: str,
        strategy_recs: Optional[Dict]
    ) -> None:
        """Print analysis results."""
        
        print("\n📈 Aggressive Assets:")
        print("-" * 40)
        if not agg_ranked.empty:
            print(agg_ranked.to_string(float_format='{:.2f}'.format))
        
        print("\n🛡️ Protective Assets:")
        print("-" * 40)
        if not prot_ranked.empty:
            print(prot_ranked.to_string(float_format='{:.2f}'.format))
        
        print("\n" + "=" * 50)
        if mode == AllocationMode.DEFENSIVE:
            print("🛡️ DEFENSIVE MODE: Negative momentum detected")
        else:
            print("📈 GROWTH MODE: All aggressive assets positive")
        
        print(f"\n🎯 Selected ETF: {selected_etf}")
        
        if strategy_recs:
            print("\n📊 Strategy Recommendations:")
            for strategy, data in strategy_recs.items():
                if 'asset' in data:
                    print(f"  • {strategy}: {data['asset']} (Score: {data['score']:.2f})")
    
    def analyze_all_strategies(
        self,
        calculation_date: Optional[date] = None
    ) -> Dict[StrategyType, SelectionResult]:
        """
        Run analysis with all available strategies.
        
        Args:
            calculation_date: Reference date
            
        Returns:
            Dictionary of strategy -> result
        """
        results = {}
        
        for strategy in StrategyType:
            results[strategy] = self.select(calculation_date, strategy)
        
        return results
    
    def get_win_probabilities(
        self,
        calculation_date: Optional[date] = None,
        months: int = 1
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate win probabilities using Monte Carlo simulation.
        
        Args:
            calculation_date: Reference date
            months: Forecast horizon
            
        Returns:
            Tuple of (win_probabilities, forecast_paths)
        """
        if not self.use_forecasting or not self.forecaster:
            return pd.Series(), pd.DataFrame()
        
        if calculation_date is None:
            calculation_date = date.today()
        
        from datetime import timedelta
        
        # Determine which universe to analyze
        agg_ranked, _ = self.momentum_analyzer.calculate_and_rank(
            self.aggressive_tickers, calculation_date
        )
        
        is_defensive = self.momentum_analyzer.is_negative_momentum(agg_ranked)
        tickers = self.protective_tickers if is_defensive else self.aggressive_tickers
        
        # Get historical momentum
        start_date = calculation_date - timedelta(days=365*2)
        hist_momentum = self.momentum_analyzer.calculate_historical_momentum(
            tickers, start_date, calculation_date
        )
        
        if hist_momentum.empty:
            return pd.Series(), pd.DataFrame()
        
        return self.forecaster.simulate_momentum_ou(hist_momentum, months)
