"""
Risk Analysis Module

Provides comprehensive risk metrics and analysis for portfolio management.

퀀트 관점:
- 리스크 관리는 수익률만큼 중요
- Sharpe Ratio, Max Drawdown, VaR 등 핵심 지표 제공
- 포트폴리오 최적화의 기초
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import MOMENTUM, RISK_FREE_RATE


@dataclass
class RiskMetrics:
    """Container for risk metrics."""

    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    beta: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None


class RiskAnalyzer:
    """
    Risk analysis and metrics calculation.

    퀀트 조언:
    - 변동성: 연간화 기준 (일별 × √252)
    - Sharpe Ratio > 1.0이면 우수, > 2.0이면 탁월
    - Max Drawdown 15% 이하가 장기 투자에 적합
    - VaR/CVaR은 꼬리 위험(tail risk) 측정
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        """
        Initialize risk analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, prices: pd.Series, method: str = "simple") -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series
            method: 'simple' or 'log'

        Returns:
            Returns series
        """
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).

        퀀트 조언:
        - 연간화: 일별 변동성 × √252
        - 주식 평균 변동성: 15-20%
        - 채권 평균 변동성: 5-10%

        Args:
            returns: Returns series
            annualize: Whether to annualize

        Returns:
            Volatility value
        """
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(MOMENTUM.TRADING_DAYS_PER_YEAR)
        return vol

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float | None = None
    ) -> float:
        """
        Calculate Sharpe Ratio.

        퀀트 조언:
        - Sharpe = (Return - Rf) / Volatility
        - > 1.0: 좋음, > 2.0: 매우 좋음, > 3.0: 탁월
        - 장기 투자 전략 평가의 핵심 지표

        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (default: class rate)

        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Annualize returns
        annual_return = returns.mean() * MOMENTUM.TRADING_DAYS_PER_YEAR
        annual_vol = self.calculate_volatility(returns, annualize=True)

        if annual_vol == 0:
            return 0.0

        return (annual_return - risk_free_rate) / annual_vol

    def calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float | None = None, target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation).

        퀀트 조언:
        - Sortino는 하방 위험만 고려 (상승 변동은 좋은 것)
        - 비대칭 수익 분포에서 Sharpe보다 적합
        - 헤지펀드 평가에 자주 사용

        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate
            target_return: Target return for downside calculation

        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        annual_return = returns.mean() * MOMENTUM.TRADING_DAYS_PER_YEAR

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return float("inf")

        downside_std = downside_returns.std() * np.sqrt(MOMENTUM.TRADING_DAYS_PER_YEAR)

        if downside_std == 0:
            return float("inf")

        return (annual_return - risk_free_rate) / downside_std

    def calculate_max_drawdown(
        self, prices: pd.Series | np.ndarray
    ) -> tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate Maximum Drawdown.

        퀀트 조언:
        - MDD = 고점에서 저점까지 최대 하락폭
        - 심리적으로 매우 중요한 지표
        - VAA 전략 목표: MDD < 15%
        - Buy & Hold S&P 500: MDD ~55% (2008-2009)

        Args:
            prices: Price series

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)

        # Calculate running maximum
        running_max = prices.expanding().max()

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Find max drawdown
        max_dd = drawdown.min()

        # Find dates
        trough_idx = drawdown.idxmin()
        peak_idx = prices.loc[:trough_idx].idxmax()

        return abs(max_dd), peak_idx, trough_idx

    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """
        Calculate Calmar Ratio (Return / Max Drawdown).

        퀀트 조언:
        - Calmar = 연환산 수익률 / MDD
        - > 1.0이면 MDD를 1년 내 회복 가능
        - 헤지펀드 목표: Calmar > 1.5

        Args:
            returns: Returns series
            prices: Price series for MDD calculation

        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * MOMENTUM.TRADING_DAYS_PER_YEAR
        max_dd, _, _ = self.calculate_max_drawdown(prices)

        if max_dd == 0:
            return float("inf")

        return annual_return / max_dd

    def calculate_var(
        self, returns: pd.Series, confidence: float = 0.95, method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.

        퀀트 조언:
        - VaR: 주어진 신뢰수준에서 최대 손실
        - 95% VaR = 5% 확률로 이보다 큰 손실 발생
        - 규제 자본 계산의 핵심 (Basel III)

        Args:
            returns: Returns series
            confidence: Confidence level
            method: 'historical' or 'parametric'

        Returns:
            VaR value (positive number representing loss)
        """
        if method == "historical":
            var = returns.quantile(1 - confidence)
        else:
            # Parametric (assumes normal distribution)
            from scipy import stats

            mean = returns.mean()
            std = returns.std()
            var = stats.norm.ppf(1 - confidence, mean, std)

        return abs(var)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        퀀트 조언:
        - CVaR = VaR을 초과하는 손실의 평균
        - VaR보다 꼬리 위험을 더 잘 포착
        - 점점 VaR 대신 CVaR 사용 추세

        Args:
            returns: Returns series
            confidence: Confidence level

        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence)
        var_threshold = -var  # Convert to return space

        # Average of returns below VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return var

        return abs(tail_returns.mean())

    def calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Beta (market sensitivity).

        퀀트 조언:
        - Beta = Cov(r, r_m) / Var(r_m)
        - β > 1: 시장보다 변동성 높음
        - β < 1: 시장보다 변동성 낮음
        - VAA 목표: β < 0.5 (시장 대비 낮은 변동성)

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark (market) returns

        Returns:
            Beta value
        """
        # Align series
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 1.0

        cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        var = aligned.iloc[:, 1].var()

        if var == 0:
            return 1.0

        return cov / var

    def calculate_all_metrics(
        self, returns: pd.Series, prices: pd.Series, benchmark_returns: pd.Series | None = None
    ) -> RiskMetrics:
        """
        Calculate all risk metrics.

        Args:
            returns: Strategy returns
            prices: Price series
            benchmark_returns: Optional benchmark returns

        Returns:
            RiskMetrics dataclass
        """
        vol = self.calculate_volatility(returns)
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, _, _ = self.calculate_max_drawdown(prices)
        calmar = self.calculate_calmar_ratio(returns, prices)
        var_95 = self.calculate_var(returns)
        cvar_95 = self.calculate_cvar(returns)

        beta = None
        tracking_error = None
        information_ratio = None

        if benchmark_returns is not None:
            beta = self.calculate_beta(returns, benchmark_returns)

            # Tracking error
            active_returns = returns - benchmark_returns
            tracking_error = self.calculate_volatility(active_returns.dropna())

            # Information ratio
            if tracking_error > 0:
                excess_return = (
                    returns.mean() - benchmark_returns.mean()
                ) * MOMENTUM.TRADING_DAYS_PER_YEAR
                information_ratio = excess_return / tracking_error

        return RiskMetrics(
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
        )

    def get_risk_report(self, metrics: RiskMetrics) -> str:
        """
        Generate human-readable risk report.

        Args:
            metrics: RiskMetrics object

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 50)
        report.append("📊 RISK ANALYSIS REPORT")
        report.append("=" * 50)

        report.append(f"\n📈 Volatility: {metrics.volatility:.1%}")
        report.append("   (Annualized standard deviation of returns)")

        report.append(f"\n⚡ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        quality = (
            "🟢 Excellent"
            if metrics.sharpe_ratio > 2
            else "🟡 Good"
            if metrics.sharpe_ratio > 1
            else "🟠 Fair"
            if metrics.sharpe_ratio > 0.5
            else "🔴 Poor"
        )
        report.append(f"   Quality: {quality}")

        report.append(f"\n📉 Max Drawdown: {metrics.max_drawdown:.1%}")
        mdd_quality = (
            "🟢 Low Risk"
            if metrics.max_drawdown < 0.15
            else "🟡 Moderate"
            if metrics.max_drawdown < 0.25
            else "🔴 High Risk"
        )
        report.append(f"   Risk Level: {mdd_quality}")

        report.append(f"\n🎯 Calmar Ratio: {metrics.calmar_ratio:.2f}")
        report.append("   (Return / Max Drawdown)")

        report.append(f"\n⚠️ VaR (95%): {metrics.var_95:.1%}")
        report.append("   (5% chance of losing more in a day)")

        report.append(f"\n💀 CVaR (95%): {metrics.cvar_95:.1%}")
        report.append("   (Expected loss when VaR is breached)")

        if metrics.beta is not None:
            report.append(f"\n📊 Beta: {metrics.beta:.2f}")
            beta_desc = (
                "more volatile than"
                if metrics.beta > 1
                else "less volatile than"
                if metrics.beta < 1
                else "same as"
            )
            report.append(f"   (Portfolio is {beta_desc} the market)")

        if metrics.information_ratio is not None:
            report.append(f"\n🎖️ Information Ratio: {metrics.information_ratio:.2f}")
            report.append("   (Active return per unit of tracking error)")

        report.append("\n" + "=" * 50)

        return "\n".join(report)
