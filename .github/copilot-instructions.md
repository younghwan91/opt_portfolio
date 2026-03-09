# Copilot Instructions

## Project Overview

This is a quantitative portfolio management system implementing the **VAA (Vigilant Asset Allocation)** strategy — a momentum-based tactical asset allocation approach. It dynamically selects ETFs monthly using momentum signals, optimizes portfolio weights via Sharpe Ratio maximization, and provides backtesting, risk analysis, and forecasting.

## Commands

```bash
# Install (uv)
make install           # uv sync --extra dev

# Run
make run               # interactive menu
make web               # Streamlit UI
python3 run.py --cli   # CLI
python3 run.py --backtest
python3 run.py --optimize

# Test
make test                                           # full suite with coverage
make test-one T=tests/test_config.py::TestAllocationConfig::test_default_validation  # single test

# Code quality
make lint              # ruff check + ruff format --check
make format            # ruff (auto-fix lint + format)
make typecheck         # mypy src/

# Cleanup
make clean             # removes __pycache__, .coverage, htmlcov/, dist/, *.db
```

## 브랜치 전략

- **`develop`** — 기능 개발 통합 브랜치. PR 머지 방식: **Squash merge** (커밋 히스토리 정리)
- **`main`** — 안정 릴리즈 브랜치. `develop` → `main` PR로만 반영

> GitHub 저장소 설정에서 `develop` 브랜치의 "Allow squash merging"만 활성화하고 나머지(merge commit, rebase)는 비활성화할 것을 권장합니다.


Data flows through layered components:

1. **Cache layer** (`src/opt_portfolio/core/cache.py`, `data_cache.py`) — DuckDB-backed incremental cache; only fetches missing date ranges from Yahoo Finance (`yfinance`). Also provides a standalone `DataCache` class in `data_cache.py`.

2. **Strategy layer** (`src/opt_portfolio/strategies/`)
   - `momentum.py` — Keller's weighted momentum formula: `12×(1M) + 4×(3M) + 2×(6M) + 1×(12M)`
   - `vaa.py` — Ranks assets within aggressive/protective universes; switches to defensive mode when absolute momentum < 0
   - `ou_process.py` — Ornstein-Uhlenbeck forecasting for mean-reversion momentum prediction

3. **Analysis layer** (`src/opt_portfolio/analysis/`)
   - `backtest.py` — Monthly walk-forward simulation with dynamic VAA selection; applies 0.1% transaction costs
   - `optimizer.py` — Grid-search over weight combinations (VAA: 20–70%, core: 5–35% each); maximizes Sharpe
   - `risk.py` — Sharpe, Sortino, max drawdown, VaR/CVaR, beta, tracking error
   - `performance.py` — CAGR, rolling returns, performance attribution

4. **Portfolio layer** (`src/opt_portfolio/core/portfolio.py`) — Tracks positions (`Position`), handles buy/sell `Transaction`s and rebalancing.

5. **UI layer** (`src/opt_portfolio/ui/`)
   - `streamlit_app.py` — Web UI with interactive Plotly charts
   - `cli.py` — Terminal menu (supports Korean/Japanese)

## Key Conventions

### Asset Universe (defined in `config.py`)
- **Aggressive tickers**: `SPY`, `EFA`, `EEM`, `AGG`
- **Protective tickers**: `LQD`, `IEF`, `SHY`
- **Core tickers**: `SPY`, `TLT`, `GLD`, `BIL`

### Default Allocation
- VAA selected ETF: 50% (`VAA_SELECTED_WEIGHT`)
- Core assets (SPY, TLT, GLD, BIL): 12.5% each
- Customizable via `AllocationConfig.from_weights(vaa, spy, tlt, gld, bil)`; weights must sum to 1.0

### Configuration Pattern
All config lives in `src/opt_portfolio/config.py` as frozen dataclasses with global singleton instances (`ASSETS`, `ALLOCATION`, `MOMENTUM`, `OU_PROCESS`, `CACHE`, `BACKTEST`, `UI`). `AllocationConfig` is the only non-frozen config (allows dynamic weight adjustment). Use `get_all_tickers()` to get the deduplicated full ticker list.

### Data Model Conventions
- Strategy output uses `SelectionResult` and `BacktestResult` dataclasses
- Risk output uses `RiskMetrics` dataclass
- Portfolio state stored in `Position` and `Transaction` dataclasses
- DuckDB stores prices in **long format** (date, ticker, price); queries return **wide format** (dates × tickers as a DataFrame)

### Risk-Free Rate
Global constant of **5%** (2025 baseline) used for all Sharpe/Sortino calculations.

### Testing
Tests live in `tests/` and match pattern `test_*.py`. Coverage is configured in `pyproject.toml` with branch coverage over `src/opt_portfolio`. The existing test suite covers configuration/validation only; strategy and backtest integration tests are not yet written.
