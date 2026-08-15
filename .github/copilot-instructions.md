# Copilot Instructions

> **정본은 저장소 루트의 `CLAUDE.md` 다.** 작업 규약·절대 규칙·벤더 함정은
> 그쪽에만 쓰고, 이 파일은 Copilot 이 빠르게 훑을 요약만 유지한다.
> 둘이 어긋나면 `CLAUDE.md` 를 따른다.

## Project Overview

두 개의 **격리된** 서브시스템이 한 저장소에 있다. 서로를 import 하지 않으며
공유하는 것은 `config.RISK_FREE_RATE` 하나뿐이다.

**1. 미국 주식 팩터 엔진 (`src/opt_portfolio/factor/`) — 현재 작업의 중심**

횡단면 팩터 전략을 검증 가능하게 만드는 엔진. 설계 원칙은 하나다 —
**조용히 틀리지 않는다.** 상장폐지 종목 포함(생존편향 제거), 표현식이 원시
테이블에 접근 못 하는 PIT 구조, 절단 시 예외, Deflated Sharpe 로 시도 횟수
정산. 팩터 158개는 선언적 DSL 로 쓰며 TTM/QoQ/YoY/가속 파생형이 자동
생성된다. 진입점은 `opt-factor` 와 `opt-factor-tui`.

**2. VAA 자산배분 (`strategies/`·`analysis/`·`core/`) — 보존 대상**

Keller 의 Vigilant Asset Allocation. 모멘텀으로 ETF 를 월간 교체하고
Sharpe 기준으로 비중을 최적화한다. 진입점은 `make run` 과 `run.py`.

**신규 팩터 코드는 `factor/` 이하에만 쓴다.** 기존 트리는 건드리지 않는다.

## Commands

```bash
# Install (uv)
make install           # uv sync --extra dev

# Run — 팩터 엔진
opt-factor optimize --store us.duckdb --config configs/x.json --space configs/s.json
opt-factor holdings --store us.duckdb --config configs/x.json   # 오늘 살 종목
opt-factor-tui      --store us.duckdb --config configs/x.json   # 운용 화면

# Run — VAA
make run               # interactive menu
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

**`main` 이 단일 작업 브랜치다.** `develop` 은 존재하지 않는다 — 1인 저장소에서
통합 브랜치는 관문 없이 단계만 늘린다. 큰 작업은 짧은 토픽 브랜치를 만들어
main 으로 머지하고, dependabot PR 은 CI 통과 후 머지한다.

관문은 브랜치가 아니라 **CI(ruff · mypy · pytest) + CodeQL** 이고, 이건 실제로
동작한다. 커밋 전에 `make lint` 를 돌린다 — `ruff check` 만 돌리면 `tests/` 의
포맷 검사가 빠져 CI 에서 깨진다.


Data flows through layered components:

1. **Cache layer** (`src/opt_portfolio/core/cache.py`) — DuckDB-backed incremental cache; only fetches missing date ranges from Yahoo Finance (`yfinance`).

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
   - `cli.py` — Terminal menu (supports Korean/Japanese). Terminal-only; there is no web UI.

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
Tests live in `tests/` (VAA) and `tests/factor/` (팩터 엔진), pattern `test_*.py`. Coverage is configured in `pyproject.toml` with branch coverage over `src/opt_portfolio`. 현재 277 tests — 팩터 엔진 쪽은 PIT 불변식·벤더 계약·과최적화 정산처럼 **틀리면 조용히 틀리는 것**을 우선 덮는다.
