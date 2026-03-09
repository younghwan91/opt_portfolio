# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2025

### Added
- Dynamic VAA backtest engine with monthly walk-forward simulation
- Sharpe Ratio grid-search optimizer (VAA: 20–70%, core: 5–35%)
- Ornstein-Uhlenbeck process forecasting for momentum prediction
- Streamlit Web UI with interactive Plotly charts
- CLI interface with Korean language support
- DuckDB-backed incremental cache (only fetches missing date ranges)
- Risk analytics: Sharpe, Sortino, VaR/CVaR, max drawdown, calmar ratio
- Performance attribution and rolling return analysis
- `AllocationConfig.from_weights()` for dynamic weight customization
- `run.py` interactive menu with 9 options

### Changed
- Full modularization into `core/`, `strategies/`, `analysis/`, `ui/`, `utils/`
- UI translated to Korean
- `CORE_WEIGHT` replaced with individual asset weights (`SPY_WEIGHT`, `TLT_WEIGHT`, `GLD_WEIGHT`, `BIL_WEIGHT`)

## [1.0.0] - 2024

### Added
- Initial VAA (Vigilant Asset Allocation) strategy implementation
- Keller's momentum formula: `12×(1M) + 4×(3M) + 2×(6M) + 1×(12M)`
- Aggressive universe: SPY, EFA, EEM, AGG
- Protective universe: LQD, IEF, SHY
- Core positions: SPY, TLT, GLD, BIL
- Yahoo Finance data fetching via `yfinance`
- MIT License

[Unreleased]: https://github.com/yourusername/opt_portfolio/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/yourusername/opt_portfolio/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/yourusername/opt_portfolio/releases/tag/v1.0.0
