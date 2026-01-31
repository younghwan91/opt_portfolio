"""
Ornstein-Uhlenbeck Process Forecasting Module

This module implements the OU process for mean-reversion modeling
and Monte Carlo simulation for momentum forecasting.

퀀트 관점:
- OU 프로세스: dX = θ(μ - X)dt + σdW
- 모멘텀 점수는 장기적으로 0 주변으로 회귀하는 특성
- Monte Carlo 시뮬레이션으로 확률적 예측 제공
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from datetime import timedelta

from ..config import OU_PROCESS, MOMENTUM


class OUForecaster:
    """
    Ornstein-Uhlenbeck process forecaster for momentum prediction.
    
    퀀트 조언:
    - OU 프로세스는 평균 회귀 시계열에 적합한 확률 모델
    - 파라미터: θ(회귀 속도), μ(장기 평균), σ(변동성)
    - 캘리브레이션: 과거 데이터로 AR(1) 회귀를 통해 추정
    """
    
    def __init__(self, num_simulations: int = OU_PROCESS.DEFAULT_SIMULATIONS):
        """
        Initialize forecaster.
        
        Args:
            num_simulations: Number of Monte Carlo paths
        """
        self.num_simulations = num_simulations
    
    def calibrate(self, series: pd.Series) -> Dict[str, float]:
        """
        Calibrate OU parameters from historical data.
        
        퀀트 조언:
        - AR(1) 회귀: X_{t+1} = α + β×X_t + ε
        - β = exp(-θ×dt) → θ = -ln(β)/dt
        - α = μ×(1-β) → μ = α/(1-β)
        - σ = std(ε)
        
        Args:
            series: Historical time series data
            
        Returns:
            Dictionary with calibrated parameters
        """
        if len(series) < MOMENTUM.MIN_DATA_POINTS:
            return {
                'theta': 0.01,
                'mu': series.mean() if len(series) > 0 else 0,
                'sigma': series.std() if len(series) > 0 else 1,
                'valid': False
            }
        
        # AR(1) regression: x_{t+1} = alpha + beta * x_t + error
        x_t = series.values[:-1]
        x_tp1 = series.values[1:]
        
        # Linear regression
        slope, intercept = np.polyfit(x_t, x_tp1, 1)
        residuals = x_tp1 - (slope * x_t + intercept)
        
        # Constrain slope for stability
        slope = max(min(slope, OU_PROCESS.SLOPE_MAX), OU_PROCESS.SLOPE_MIN)
        
        # Calculate OU parameters
        # θ = -ln(β) where β is the slope
        theta = -np.log(slope)
        
        # Constrain theta for numerical stability
        theta = max(min(theta, OU_PROCESS.THETA_MAX), OU_PROCESS.THETA_MIN)
        
        # μ = α / (1 - β)
        if abs(1 - slope) > 1e-6:
            mu = intercept / (1 - slope)
        else:
            mu = series.mean()
        
        sigma = np.std(residuals)
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'slope': slope,
            'intercept': intercept,
            'valid': True
        }
    
    def forecast(
        self, 
        series: pd.Series, 
        months: int = 1
    ) -> float:
        """
        Forecast future value using analytical OU solution.
        
        퀀트 조언:
        - 해석적 해: E[X_{t+T}] = μ + (X_t - μ)×exp(-θT)
        - 평균 회귀: 현재값이 평균보다 높으면 하락 예측, 낮으면 상승 예측
        
        Args:
            series: Historical series
            months: Months ahead to forecast
            
        Returns:
            Expected value at forecast horizon
        """
        if len(series) < MOMENTUM.MIN_DATA_POINTS:
            return series.iloc[-1] if len(series) > 0 else 0
        
        params = self.calibrate(series)
        current_val = series.iloc[-1]
        
        # Time horizon in trading days
        T = months * MOMENTUM.TRADING_DAYS_PER_MONTH
        
        # Analytical solution: E[X_{t+T}] = mu + (X_t - mu) * exp(-theta * T)
        expected_val = params['mu'] + (current_val - params['mu']) * np.exp(-params['theta'] * T)
        
        return expected_val
    
    def forecast_delta(self, series: pd.Series, months: int = 1) -> float:
        """
        Calculate expected change (delta) over forecast horizon.
        
        퀀트 조언:
        - Delta = Forecast - Current
        - 양수면 모멘텀 개선 예상, 음수면 악화 예상
        - 트렌드 변화를 포착하는 데 유용
        
        Args:
            series: Historical series
            months: Months ahead
            
        Returns:
            Expected change
        """
        current = series.iloc[-1] if len(series) > 0 else 0
        forecast = self.forecast(series, months)
        return forecast - current
    
    def simulate(
        self,
        series: pd.Series,
        months: int = 1,
        return_paths: bool = False
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        """
        Run Monte Carlo simulation for OU process.
        
        퀀트 조언:
        - Monte Carlo: 다수의 랜덤 경로 생성 후 통계 계산
        - 불확실성 정량화 (평균뿐 아니라 분포 확인)
        - 1000회 시뮬레이션은 수렴에 충분
        
        Args:
            series: Historical series
            months: Forecast horizon
            return_paths: Whether to return simulation paths
            
        Returns:
            Tuple of (mean, std, optional paths)
        """
        if len(series) < MOMENTUM.MIN_DATA_POINTS:
            current = series.iloc[-1] if len(series) > 0 else 0
            return current, 0.0, None
        
        params = self.calibrate(series)
        forecast_days = months * MOMENTUM.TRADING_DAYS_PER_MONTH
        dt = 1.0  # Time step (1 day)
        
        # Initialize simulation paths
        current_val = series.iloc[-1]
        sim_paths = np.zeros((self.num_simulations, forecast_days))
        sim_paths[:, 0] = current_val
        
        theta = params['theta']
        mu = params['mu']
        sigma = params['sigma']
        
        # Simulate paths
        for t in range(1, forecast_days):
            noise = np.random.normal(0, 1, self.num_simulations)
            # OU Update: dX = theta * (mu - X) * dt + sigma * dW
            dx = theta * (mu - sim_paths[:, t-1]) * dt + sigma * noise
            sim_paths[:, t] = sim_paths[:, t-1] + dx
        
        final_values = sim_paths[:, -1]
        mean_forecast = np.mean(final_values)
        std_forecast = np.std(final_values)
        
        if return_paths:
            return mean_forecast, std_forecast, sim_paths
        return mean_forecast, std_forecast, None
    
    def simulate_momentum_ou(
        self,
        momentum_df: pd.DataFrame,
        months: int = 1
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Simulate future momentum for multiple assets and calculate win probabilities.
        
        퀀트 조언:
        - 자산별 OU 시뮬레이션 후 "승리 확률" 계산
        - 각 시뮬레이션에서 가장 높은 모멘텀을 가진 자산이 "승리"
        - 확률 기반 의사결정에 유용
        
        Args:
            momentum_df: DataFrame with momentum scores for each asset
            months: Forecast horizon
            
        Returns:
            Tuple of (win_probabilities Series, forecast_df)
        """
        if momentum_df.empty:
            return pd.Series(), pd.DataFrame()
        
        print(f"🎲 Simulating momentum (OU Process) for next {months} months ({self.num_simulations} paths)...")
        
        forecast_days = months * MOMENTUM.TRADING_DAYS_PER_MONTH
        dt = 1.0
        
        final_scores = {ticker: [] for ticker in momentum_df.columns}
        mean_paths = {}
        
        # Create future dates
        last_date = momentum_df.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='B'
        )
        
        for ticker in momentum_df.columns:
            series = momentum_df[ticker].dropna()
            
            if len(series) < MOMENTUM.MIN_DATA_POINTS:
                print(f"⚠️ Not enough data for {ticker}, skipping simulation.")
                continue
            
            params = self.calibrate(series)
            theta = params['theta']
            mu = params['mu']
            sigma = params['sigma']
            
            current_val = series.iloc[-1]
            sim_paths = np.zeros((self.num_simulations, forecast_days))
            sim_paths[:, 0] = current_val
            
            for t in range(1, forecast_days):
                noise = np.random.normal(0, 1, self.num_simulations)
                dx = theta * (mu - sim_paths[:, t-1]) * dt + sigma * noise
                sim_paths[:, t] = sim_paths[:, t-1] + dx
            
            final_scores[ticker] = sim_paths[:, -1]
            mean_paths[ticker] = np.mean(sim_paths, axis=0)
        
        # Calculate win probabilities
        wins = {ticker: 0 for ticker in momentum_df.columns}
        valid_tickers = [
            t for t in momentum_df.columns
            if t in final_scores and len(final_scores[t]) > 0
        ]
        
        if not valid_tickers:
            return pd.Series(), pd.DataFrame()
        
        # Convert to array [num_sims, num_tickers]
        all_final_scores = np.array([final_scores[t] for t in valid_tickers]).T
        
        # Find winner for each simulation
        winners_indices = np.argmax(all_final_scores, axis=1)
        
        for idx in winners_indices:
            winner_ticker = valid_tickers[idx]
            wins[winner_ticker] += 1
        
        probs = pd.Series(wins) / self.num_simulations
        probs = probs.sort_values(ascending=False)
        
        # Create forecast DataFrame
        df_forecast = pd.DataFrame(mean_paths, index=future_dates)
        
        return probs, df_forecast
    
    def get_confidence_interval(
        self,
        series: pd.Series,
        months: int = 1,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for forecast.
        
        퀀트 조언:
        - 신뢰구간은 예측의 불확실성을 정량화
        - 95% 신뢰구간이 표준, 리스크 관리에 중요
        
        Args:
            series: Historical series
            months: Forecast horizon
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, mean, upper_bound)
        """
        mean, std, paths = self.simulate(series, months, return_paths=True)
        
        if paths is None:
            return mean, mean, mean
        
        alpha = 1 - confidence
        lower = np.percentile(paths[:, -1], alpha/2 * 100)
        upper = np.percentile(paths[:, -1], (1 - alpha/2) * 100)
        
        return lower, mean, upper
