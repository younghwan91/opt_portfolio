# 🚀 Optimal Portfolio Management System

A professional-grade quantitative portfolio management system implementing the **Vigilant Asset Allocation (VAA)** strategy with advanced **Ornstein-Uhlenbeck (OU) process forecasting**, automated rebalancing, and comprehensive risk analytics.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Strategy Overview](#-strategy-overview)
- [Quant Professional Insights](#-quant-professional-insights)
- [API Reference](#-api-reference)
- [Performance](#-performance)
- [Contributing](#-contributing)

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 🔍 **VAA Selection** | Automated ETF selection based on multi-period momentum analysis |
| 🔮 **OU Forecasting** | Mean-reversion modeling with Monte Carlo simulation |
| ⚡ **Smart Caching** | DuckDB-powered incremental data fetching |
| ⚖️ **Auto Rebalancing** | Integer share optimization with cash flow management |
| 📊 **Risk Analytics** | Sharpe, Sortino, VaR, CVaR, Max Drawdown, and more |
| 📈 **Backtesting** | Multi-strategy comparison with transaction costs |
| 🌐 **Web UI** | Interactive Streamlit dashboard with Plotly charts |
| 💻 **CLI** | Full-featured command-line interface |

### Advanced Analytics

- **Multi-Strategy Comparison**: Current, Forecast 1M/3M/6M, Delta (Momentum Velocity)
- **Win Probability Calculation**: Monte Carlo-based probability of being the best performer
- **Regime Analysis**: Up/Down market capture ratios
- **Drawdown Analysis**: Top-N drawdown periods with recovery times
- **Performance Attribution**: Year-by-year and market regime breakdown

---

## 📁 Project Structure

```
opt_portfolio/
├── src/opt_portfolio/          # Main package
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Configuration & constants
│   │
│   ├── core/                  # Core modules
│   │   ├── cache.py           # DuckDB caching system
│   │   └── portfolio.py       # Portfolio management
│   │
│   ├── strategies/            # Trading strategies
│   │   ├── vaa.py            # VAA strategy implementation
│   │   ├── momentum.py       # Momentum calculations
│   │   └── ou_process.py     # OU process forecasting
│   │
│   ├── analysis/              # Analytics modules
│   │   ├── backtest.py       # Backtesting engine
│   │   ├── risk.py           # Risk metrics
│   │   └── performance.py    # Performance analysis
│   │
│   ├── ui/                    # User interfaces
│   │   ├── streamlit_app.py  # Web UI
│   │   └── cli.py            # Command-line interface
│   │
│   └── utils/                 # Utilities
│       ├── helpers.py        # Helper functions
│       └── visualization.py  # Chart utilities
│
├── tests/                     # Test suite
├── docs/                      # Documentation
├── run.py                     # Main entry point
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/opt_portfolio.git
cd opt_portfolio
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -e .
# or for development
pip install -e ".[dev]"
```

4. **Verify installation:**
```bash
python run.py
```

---

## 🚀 Quick Start

### Option 1: Web UI (Recommended)

```bash
python run.py --web
# or
streamlit run src/opt_portfolio/ui/streamlit_app.py
```

### Option 2: Command Line Interface

```bash
python run.py --cli
```

### Option 3: Python API

```python
from opt_portfolio.strategies.vaa import VAAStrategy
from opt_portfolio.core.portfolio import Portfolio
from opt_portfolio.analysis.backtest import BacktestEngine

# Run VAA Analysis
vaa = VAAStrategy(use_forecasting=True)
result = vaa.select()
print(f"Selected ETF: {result.selected_etf}")
print(f"Mode: {'Defensive' if result.is_defensive else 'Growth'}")

# Calculate win probabilities
win_probs, forecast = vaa.get_win_probabilities(months=1)
print(f"Win Probabilities:\n{win_probs}")
```

### Legacy Interfaces (Still Available)

- **VAA analysis only**: `python vaa_agg.py`
- **Rebalancing calculator**: `python rebalance.py`
- **Backtest Comparison**: `python backtest_comparison.py`

---

## 📊 Strategy Overview

### VAA (Vigilant Asset Allocation)

VAA is a tactical asset allocation strategy developed by **Wouter Keller** (2017).

#### Asset Universes

| Universe | Assets | Purpose |
|----------|--------|---------|
| **Aggressive** | SPY, EFA, EEM, AGG | Growth during bull markets |
| **Protective** | LQD, IEF, SHY | Capital preservation during corrections |
| **Core Holdings** | SPY, TLT, GLD, BIL | Permanent strategic allocation |

#### Target Allocation

```
┌─────────────────────────────────────────┐
│                                         │
│    ┌─────────────────────┐              │
│    │  VAA Selected ETF   │    50%       │
│    │    (Tactical)       │              │
│    └─────────────────────┘              │
│                                         │
│    ┌─────┬─────┬─────┬─────┐            │
│    │ SPY │ TLT │ GLD │ BIL │  12.5% each│
│    │     │     │     │     │            │
│    └─────┴─────┴─────┴─────┘            │
│          (Core Holdings)                │
│                                         │
└─────────────────────────────────────────┘
```

#### Momentum Formula

The weighted momentum score formula:

```
Momentum Score = 12 × r_1m + 4 × r_3m + 2 × r_6m + 1 × r_12m
```

Where `r_nm` = n-month return (%)

#### Selection Logic

```python
IF any(Aggressive Momentum < 0):
    Mode = DEFENSIVE
    Select = argmax(Protective Momentum)
ELSE:
    Mode = GROWTH
    Select = argmax(Aggressive Momentum)
```

### 🔮 Advanced Forecasting & Backtesting

The system now includes a sophisticated forecasting engine:

| Strategy | Description | 15-Year Return |
|----------|-------------|----------------|
| **Standard VAA** | Selects asset with highest *current* score | **+114.6%** |
| **Forecast (1-Month)** | Selects asset with highest *predicted* score next month | **+173.7%** |
| **Velocity (Delta)** | Selects asset with highest *increase* in momentum | **+201.3%** |
| **Forecast (3-Month)** | Selects asset with highest *predicted* score in 3 months | **+238.8%** |
| **Forecast (6-Month)** | Selects asset with highest *predicted* score in 6 months | **+242.2%** |

*Note: Past performance does not guarantee future results.*

---

## 🎓 Quant Professional Insights

### 1. 모멘텀의 학술적 배경 (Academic Foundation of Momentum)

모멘텀은 학술적으로 가장 강력하게 검증된 시장 이상현상(market anomaly) 중 하나입니다.

> **"Winners continue to win, losers continue to lose"** - Jegadeesh & Titman (1993)

**VAA의 가중치 (12, 4, 2, 1) 근거:**
- 모멘텀의 반감기(half-life)는 약 3-6개월
- 단기 모멘텀에 높은 가중치 → 빠른 시장 반응
- 장기 모멘텀 포함 → 노이즈 필터링

### 2. OU 프로세스 (Ornstein-Uhlenbeck Process)

모멘텀 점수는 장기적으로 0 주변으로 회귀하는 경향이 있습니다.

```
dX_t = θ(μ - X_t)dt + σdW_t
```

| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| θ (theta) | Mean reversion speed | 0.001 - 0.1 |
| μ (mu) | Long-term mean | ~ 0 |
| σ (sigma) | Volatility | Asset-dependent |

**캘리브레이션 (Calibration):**
AR(1) 회귀를 통해 파라미터 추정:
- `β = e^(-θ)`
- `α = μ(1 - β)`

### 3. 리밸런싱 최적화 (Rebalancing Optimization)

**정수 주식 제약 (Integer Constraint):**
- 완벽한 목표 배분은 불가능
- 우선순위: 큰 편차부터 교정
- 매도 후 매수 순서로 현금 흐름 최적화

**권장 리밸런싱 주기:**

| 주기 | 장점 | 단점 |
|------|------|------|
| 일별 | 최적 추적 | 거래비용 과다 |
| 주별 | 균형 | 노이즈 거래 |
| **월별** | **비용 효율적** | **약간의 추적 오차** |
| 분기별 | 최소 비용 | 큰 편차 가능 |

### 4. 리스크 지표 해석 (Risk Metrics Interpretation)

| 지표 | 좋음 (Good) | 보통 (Average) | 주의 (Warning) |
|------|------|------|------|
| Sharpe Ratio | > 2.0 | 1.0 - 2.0 | < 1.0 |
| Max Drawdown | < 15% | 15-25% | > 25% |
| Calmar Ratio | > 1.5 | 1.0 - 1.5 | < 1.0 |
| Win Rate | > 60% | 50-60% | < 50% |

### 5. 백테스트 주의사항 (Backtesting Caveats)

⚠️ **과적합 (Overfitting) 경고:**
- In-sample 성과 ≠ Out-of-sample 성과
- 파라미터 최적화 → 과적합 위험
- Walk-forward 분석 권장

⚠️ **Survivorship Bias:**
- 상장폐지된 종목 누락 → 성과 과대평가
- ETF는 상대적으로 안전

⚠️ **Look-Ahead Bias:**
- 미래 데이터 사용 → 비현실적 성과
- 월말 가격만 사용 (조정 종가)

### 6. 실전 적용 가이드 (Practical Implementation Guide)

**최소 자본금 권장:**
```
$10,000 이상 (allocation error < 3%)
$50,000 이상 (allocation error < 1%)
```

**거래 비용:**
- ETF 스프레드: ~0.01%
- 커미션: $0 (대부분의 브로커)
- 총 예상 비용: ~0.1% per rebalance

**세금 고려:**
- 월별 리밸런싱 → 단기 양도소득
- 세금 이연 계좌 활용 권장 (IRA, 401k 등)

---

## 📚 API Reference

### VAAStrategy

```python
from opt_portfolio.strategies.vaa import VAAStrategy

vaa = VAAStrategy(
    aggressive_tickers=['SPY', 'EFA', 'EEM', 'AGG'],
    protective_tickers=['LQD', 'IEF', 'SHY'],
    use_cache=True,
    use_forecasting=True
)

# Run selection
result = vaa.select(calculation_date=date.today())

# Get win probabilities
win_probs, forecast_df = vaa.get_win_probabilities(months=1)
```

### Portfolio

```python
from opt_portfolio.core.portfolio import Portfolio

portfolio = Portfolio.from_dict({'SPY': 100, 'TLT': 50})
portfolio.update_prices()

# Get current allocation
allocation = portfolio.get_allocation()

# Calculate rebalance
recommendations = portfolio.calculate_rebalance(
    selected_etf='AGG',
    additional_cash=10000
)
```

### BacktestEngine

```python
from opt_portfolio.analysis.backtest import BacktestEngine

engine = BacktestEngine(
    initial_capital=10000,
    transaction_cost=0.001  # 0.1%
)

results = engine.run_vaa_backtest(years=15)
engine.plot_results(results)
```

### RiskAnalyzer

```python
from opt_portfolio.analysis.risk import RiskAnalyzer

analyzer = RiskAnalyzer(risk_free_rate=0.05)
metrics = analyzer.calculate_all_metrics(returns=monthly_returns)
print(analyzer.get_risk_report(metrics))
```

---

## 🛠️ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.24.0 | Numerical computations |
| **pandas** | ≥2.0.0 | Data manipulation and analysis |
| **yfinance** | ≥0.2.36 | Real-time financial data |
| **streamlit** | ≥1.28.0 | Web UI framework |
| **plotly** | ≥5.18.0 | Interactive charts |
| **duckdb** | ≥0.9.0 | Fast columnar caching |
| **scipy** | ≥1.11.0 | Statistical analysis |

---

## 🚨 Important Notes

- **📊 Data Source**: Uses Yahoo Finance API for real-time pricing
- **🕐 Market Hours**: Best results during market hours for accurate pricing
- **🔄 Rebalancing Frequency**: Recommend monthly rebalancing
- **⚠️ Risk Disclaimer**: This is educational software, not financial advice

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- 🐛 Bug fixes
- ✨ New features  
- 📚 Documentation improvements
- 🧪 Additional testing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Investing involves risk of loss
- Always consult a qualified financial advisor
- The authors are not responsible for any financial losses

---

## 📜 License

This project is open source and available under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Wouter Keller for the VAA strategy framework
- Yahoo Finance for market data
- The open-source community for amazing tools

---

*Built with ❤️ for quantitative investors*

**🎯 Ready to optimize your portfolio?** Start with `python run.py` and choose your preferred interface!
