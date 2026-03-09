"""
Configuration and Constants for Portfolio Management System

This module contains all configurable parameters, default settings,
and strategy constants used throughout the application.

퀀트 관점 조언:
- 리밸런싱 주기는 월 1회가 최적 (거래비용과 수익률의 균형)
- 모멘텀 가중치 (12, 4, 2, 1)는 Keller의 VAA 논문 기반
- SHY를 안전자산으로 선택한 이유: 금리 민감도가 가장 낮음
"""

from dataclasses import dataclass, field
from enum import Enum


class AllocationMode(Enum):
    """Portfolio allocation mode based on market conditions."""

    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"


class StrategyType(Enum):
    """Available momentum strategy types."""

    CURRENT = "current"
    FORECAST_1M = "forecast_1m"
    FORECAST_3M = "forecast_3m"
    FORECAST_6M = "forecast_6m"
    DELTA = "delta"


@dataclass(frozen=True)
class AssetUniverse:
    """
    Asset universe configuration for VAA strategy.

    퀀트 조언:
    - Aggressive 자산: 위험 자산으로 상승장에서 수익 추구
    - Protective 자산: 하락장 방어용, 변동성 낮은 채권 중심
    - Core Holdings: 영구 포트폴리오 철학 기반 (Ray Dalio's All Weather)
    """

    # VAA Aggressive Universe (위험 추구 자산군)
    AGGRESSIVE_TICKERS: tuple = ("SPY", "EFA", "EEM", "AGG")
    AGGRESSIVE_NAMES: dict[str, str] = field(
        default_factory=lambda: {
            "SPY": "S&P 500 (미국 대형주)",
            "EFA": "MSCI EAFE (선진국 ex-미국)",
            "EEM": "MSCI Emerging Markets (신흥국)",
            "AGG": "US Aggregate Bond (미국 종합 채권)",
        }
    )

    # VAA Protective Universe (방어 자산군)
    PROTECTIVE_TICKERS: tuple = ("LQD", "IEF", "SHY")
    PROTECTIVE_NAMES: dict[str, str] = field(
        default_factory=lambda: {
            "LQD": "Investment Grade Corporate (투자등급 회사채)",
            "IEF": "7-10 Year Treasury (중기 국채)",
            "SHY": "1-3 Year Treasury (단기 국채, 현금 대용)",
        }
    )

    # Core Holdings (영구 보유 자산군 - All Weather 기반)
    CORE_TICKERS: tuple = ("SPY", "TLT", "GLD", "BIL")
    CORE_NAMES: dict[str, str] = field(
        default_factory=lambda: {
            "SPY": "S&P 500 (주식)",
            "TLT": "20+ Year Treasury (장기 국채)",
            "GLD": "Gold (금, 인플레이션 헤지)",
            "BIL": "T-Bills (현금성 자산)",
        }
    )


@dataclass
class AllocationConfig:
    """
    Target allocation percentages - now optimizable via Sharpe ratio optimization.

    퀀트 조언:
    - 기본값 50%/12.5%는 Keller 원본 기반
    - 최적화 시 Sharpe Ratio 기준으로 비율 조정 가능
    - Core 자산 비중은 개별 조정 가능 (SPY, TLT, GLD, BIL)
    """

    VAA_SELECTED_WEIGHT: float = 0.50  # VAA로 선택된 ETF 비중

    # Core assets - 이제 개별 조정 가능
    SPY_WEIGHT: float = 0.125
    TLT_WEIGHT: float = 0.125
    GLD_WEIGHT: float = 0.125
    BIL_WEIGHT: float = 0.125

    # Error tolerance for rebalancing (%)
    REBALANCE_THRESHOLD: float = 5.0  # 5% 이상 벗어나면 리밸런싱
    ACCEPTABLE_ERROR: float = 2.0  # 2% 이내면 양호

    @property
    def target_allocations(self) -> dict[str, float]:
        """Generate target allocation percentages."""
        return {
            "selected": self.VAA_SELECTED_WEIGHT,
            "SPY": self.SPY_WEIGHT,
            "TLT": self.TLT_WEIGHT,
            "GLD": self.GLD_WEIGHT,
            "BIL": self.BIL_WEIGHT,
        }

    @property
    def core_weights(self) -> dict[str, float]:
        """Get core asset weights as dictionary."""
        return {
            "SPY": self.SPY_WEIGHT,
            "TLT": self.TLT_WEIGHT,
            "GLD": self.GLD_WEIGHT,
            "BIL": self.BIL_WEIGHT,
        }

    def validate(self) -> bool:
        """Validate that allocations sum to 1.0."""
        total = (
            self.VAA_SELECTED_WEIGHT
            + self.SPY_WEIGHT
            + self.TLT_WEIGHT
            + self.GLD_WEIGHT
            + self.BIL_WEIGHT
        )
        return abs(total - 1.0) < 0.001

    @classmethod
    def from_weights(
        cls, vaa: float, spy: float, tlt: float, gld: float, bil: float
    ) -> "AllocationConfig":
        """Create config from explicit weights."""
        config = cls(
            VAA_SELECTED_WEIGHT=vaa, SPY_WEIGHT=spy, TLT_WEIGHT=tlt, GLD_WEIGHT=gld, BIL_WEIGHT=bil
        )
        if not config.validate():
            raise ValueError(f"Weights must sum to 1.0, got {vaa + spy + tlt + gld + bil}")
        return config


@dataclass(frozen=True)
class MomentumConfig:
    """
    Momentum calculation parameters.

    퀀트 조언 (Keller's VAA 논문 기반):
    - 가중치 12, 4, 2, 1은 단기 모멘텀에 더 높은 가중치 부여
    - 이는 모멘텀의 반감기(half-life)가 약 3-6개월이라는 연구 결과 반영
    - 252 거래일 = 1년, 21 거래일 = 1개월 (시장 컨벤션)
    """

    # Period definitions (in trading days)
    TRADING_DAYS_PER_MONTH: int = 21
    TRADING_DAYS_PER_YEAR: int = 252

    PERIOD_1M_DAYS: int = 21
    PERIOD_3M_DAYS: int = 63
    PERIOD_6M_DAYS: int = 126
    PERIOD_12M_DAYS: int = 252

    # Momentum score weights (Keller's VAA formula)
    WEIGHT_1M: int = 12
    WEIGHT_3M: int = 4
    WEIGHT_6M: int = 2
    WEIGHT_12M: int = 1

    # Minimum data requirements
    MIN_DATA_POINTS: int = 30
    CALIBRATION_WINDOW: int = 60  # 60일 데이터로 OU 프로세스 캘리브레이션


@dataclass(frozen=True)
class OUProcessConfig:
    """
    Ornstein-Uhlenbeck process configuration for forecasting.

    퀀트 조언:
    - OU 프로세스는 평균 회귀(mean reversion) 모델링에 적합
    - 모멘텀 점수는 장기적으로 0 주변으로 회귀하는 경향
    - theta 제한(0.001-0.1)은 모델 안정성을 위함
    - 시뮬레이션 1000회는 Monte Carlo 수렴에 충분
    """

    # Mean reversion speed bounds
    THETA_MIN: float = 0.001
    THETA_MAX: float = 0.1

    # Slope bounds for stability
    SLOPE_MIN: float = 0.001
    SLOPE_MAX: float = 0.999

    # Monte Carlo simulation parameters
    DEFAULT_SIMULATIONS: int = 1000
    DEFAULT_FORECAST_MONTHS: int = 1


@dataclass(frozen=True)
class CacheConfig:
    """
    Data caching configuration.

    퀀트 조언:
    - DuckDB는 시계열 데이터에 최적화된 열 기반 저장소
    - 400일 버퍼는 12개월 수익률 계산 + 여유분
    - 증분 업데이트로 API 호출 최소화 (yfinance rate limit 대응)
    """

    DB_PATH: str = ".cache/market_data.duckdb"
    DATA_BUFFER_DAYS: int = 400  # 12개월 모멘텀 계산을 위한 버퍼
    DEFAULT_CACHE_EXPIRY_DAYS: int = 30


@dataclass
class BacktestConfig:
    """
    Backtesting configuration.

    퀀트 조언:
    - 15년 백테스트는 최소 2번의 경제 사이클 포함 (2008 금융위기, 2020 코로나)
    - 월별 리밸런싱이 비용 효율적 (일별은 거래비용 과다)
    - 시작 자본 $10,000은 계산 편의를 위함 (실제 수익률에 영향 없음)
    """

    DEFAULT_YEARS: int = 15
    INITIAL_CAPITAL: float = 10000.0
    REBALANCE_FREQUENCY: str = "monthly"  # 'daily', 'weekly', 'monthly'
    TRANSACTION_COST: float = 0.001  # 0.1% 거래비용 (ETF 평균)

    # Optimization settings
    OPTIMIZATION_WEIGHT_MIN: float = 0.05  # 최소 5% 비중
    OPTIMIZATION_WEIGHT_MAX: float = 0.70  # 최대 70% 비중
    OPTIMIZATION_STEP: float = 0.05  # 5% 단위로 최적화


@dataclass(frozen=True)
class UIConfig:
    """UI configuration settings."""

    PAGE_TITLE: str = "Optimal Portfolio Management System"
    PAGE_ICON: str = "🚀"
    LAYOUT: str = "wide"

    # Chart settings
    CHART_HEIGHT: int = 400
    CHART_WIDTH: int = 800

    # Color scheme
    COLORS = {
        "positive": "#00C851",
        "negative": "#ff4444",
        "neutral": "#33b5e5",
        "warning": "#ffbb33",
        "primary": "#2196F3",
        "secondary": "#757575",
    }


# Global configuration instances
ASSETS = AssetUniverse()
ALLOCATION = AllocationConfig()
MOMENTUM = MomentumConfig()
OU_PROCESS = OUProcessConfig()
CACHE = CacheConfig()
BACKTEST = BacktestConfig()
UI = UIConfig()


# Risk-Free Rate (for Sharpe Ratio calculations)
# 퀀트 조언: 현재 미국 3개월 T-Bill 금리 기준, 정기적 업데이트 필요
RISK_FREE_RATE = 0.05  # 5% (2025년 기준)


def get_all_tickers() -> list[str]:
    """Get all unique tickers across all universes."""
    all_tickers = set(ASSETS.AGGRESSIVE_TICKERS)
    all_tickers.update(ASSETS.PROTECTIVE_TICKERS)
    all_tickers.update(ASSETS.CORE_TICKERS)
    return list(all_tickers)
