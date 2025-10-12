# 🚀 Portfolio Management System

A comprehensive portfolio management system implementing the **Vigilant Asset Allocation (VAA)** strategy with automated rebalancing capabilities and intelligent optimization.

## 🎯 Features

- **🔍 VAA ETF Selection**: Automatically selects the optimal ETF based on momentum analysis across multiple time periods
- **⚖️ Strategic Allocation**: 50% to VAA-selected ETF, 12.5% each to core holdings (SPY, TLT, GLD, BIL)
- **🧮 Smart Rebalancing**: Calculates exact buy/sell orders with cash flow optimization
- **💻 Multiple Interfaces**: Command-line interface and modern web UI
- **📊 Real-time Data**: Uses Yahoo Finance for current market prices and historical performance
- **📈 Advanced Analytics**: Allocation error analysis and optimization quality metrics

## 🏗️ Installation

### Prerequisites
- Python 3.13+
- pip package manager

### Setup
1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd opt_portfolio
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python main.py
```

## 🚀 Usage

### Quick Start
Run the main menu and choose your interface:
```bash
python main.py
```

### Available Interfaces

#### 1. 🌐 **Web UI (Recommended)**
Launch the interactive Streamlit web interface:
```bash
streamlit run portfolio_ui.py
```
*Features: Real-time portfolio input, visual analytics, detailed transaction breakdown*

#### 2. 💻 **Command Line Interface**
For complete portfolio management in terminal:
```bash
python integrated_portfolio.py
```
*Features: Step-by-step guidance, portfolio analysis, interactive rebalancing*

#### 3. 🔧 **Individual Components**
- **VAA analysis only**: `python vaa_agg.py`
- **Portfolio calculator only**: `python port_ratio_calculator.py`
- **Rebalancing calculator**: `python rebalance.py`

## 📊 Portfolio Strategy

### Core Strategy
The system implements a **strategic-tactical hybrid allocation**:

| Asset Class | Allocation | Purpose |
|-------------|------------|---------|
| **VAA Selected ETF** | **50%** | Tactical allocation based on momentum |
| **SPY** (S&P 500) | **12.5%** | Core equity exposure |
| **TLT** (Long Treasury) | **12.5%** | Interest rate hedge |
| **GLD** (Gold) | **12.5%** | Inflation protection |
| **BIL** (Short Treasury) | **12.5%** | Cash equivalent/liquidity |

### 🎯 VAA Selection Process

1. **📊 Data Collection**: Gathers 1, 3, 6, and 12-month performance data
2. **🔥 Aggressive Analysis**: Analyzes SPY, EFA, EEM, AGG for growth potential
3. **🛡️ Protective Analysis**: Evaluates LQD, IEF, SHY for capital preservation
4. **🧮 Momentum Scoring**: Calculates weighted momentum scores (1m×12 + 3m×4 + 6m×2 + 12m×1)
5. **🚦 Decision Logic**: 
   - **Defensive Mode**: Selects protective assets if ANY aggressive asset shows negative momentum
   - **Growth Mode**: Selects top aggressive asset when all show positive momentum

### 🔄 Rebalancing Engine

The system features an **intelligent optimization engine** that:

- **💸 Maximizes Sales**: Identifies excess positions to generate rebalancing cash
- **🎯 Optimizes Purchases**: Prioritizes investments to minimize allocation errors
- **💰 Cash Management**: Efficiently utilizes available cash including additional investments
- **📊 Error Analysis**: Provides detailed allocation accuracy metrics

## 📁 Architecture

```
opt_portfolio/
├── main.py                     # 🚪 Entry point with menu system
├── integrated_portfolio.py     # 💻 Complete CLI portfolio management
├── portfolio_ui.py            # 🌐 Streamlit web interface
├── vaa_agg.py                # 📈 VAA momentum analysis engine
├── port_ratio_calculator.py   # 📊 Portfolio composition calculator
├── rebalance.py              # ⚖️ Rebalancing optimization engine
├── requirements.txt          # 📦 Python dependencies
├── pyproject.toml           # ⚙️ Project configuration
└── README.md                # 📚 Documentation
```

## 💡 Example Usage

### Scenario: Rebalancing with Additional Investment

```python
from rebalance import calculate_rebalance, print_rebalance_report

# Your current portfolio
current_portfolio = {
    'EEM': 27,    # Current VAA selection
    'SPY': 0,     # Need to buy
    'TLT': 3,     # May need adjustment
    'GLD': 1,     # May need adjustment  
    'BIL': 3      # May need adjustment
}

# Calculate optimal rebalancing with $1000 additional investment
recommendations = calculate_rebalance(
    current_portfolio, 
    selected_etf="EEM",    # From VAA analysis
    additional_cash=1000
)

# Display detailed recommendations
print_rebalance_report(recommendations)
```

### Expected Output Features:
- 📊 **Current vs Target Allocation Analysis**
- 💸 **Optimized Buy/Sell Transactions** 
- 💰 **Cash Flow Optimization**
- 🎯 **Allocation Error Metrics**
- ✅ **Optimization Quality Assessment**

## 📈 Advanced Features

### 🔍 **Allocation Error Analysis**
- Tracks percentage deviation from target allocations
- Provides optimization quality scoring
- Suggests improvements for better allocation accuracy

### 💰 **Cash Flow Optimization**
- Maximizes use of available cash (additional + sales proceeds)
- Minimizes remaining uninvested cash
- Calculates optimal transaction sequences

### 📊 **Performance Metrics**
- Real-time portfolio valuation
- Historical momentum scoring
- Transaction cost analysis
- Allocation efficiency tracking

## 🛠️ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥2.3.3 | Numerical computations |
| **pandas** | ≥2.3.2 | Data manipulation and analysis |
| **yfinance** | ≥0.2.66 | Real-time financial data |
| **streamlit** | ≥1.28.0 | Web UI framework |
| **selenium** | ≥4.35.0 | Web scraping (if needed) |
| **python-dateutil** | ≥2.8.2 | Date manipulation utilities |

## 🚨 Important Notes

- **📊 Data Source**: Uses Yahoo Finance API for real-time pricing
- **🕐 Market Hours**: Best results during market hours for accurate pricing
- **🔄 Rebalancing Frequency**: Recommend monthly or quarterly rebalancing
- **⚠️ Risk Disclaimer**: This is educational software, not financial advice

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- 🐛 Bug fixes
- ✨ New features  
- 📚 Documentation improvements
- 🧪 Additional testing

## 📜 License

This project is open source and available under the **MIT License**.

---

**🎯 Ready to optimize your portfolio?** Start with `python main.py` and choose your preferred interface!
