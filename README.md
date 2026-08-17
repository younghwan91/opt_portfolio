# opt_portfolio

**English** · [한국어](README.ko.md)

**US equity factor engine + tactical asset allocation (VAA) backtester.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

Two independent subsystems live in this repository.

| | **Factor engine** (`factor/`) | **VAA allocation** (`strategies/`·`analysis/`) |
|---|---|---|
| Scope | US single stocks (20,931 tickers, 1997–2026) | 7–11 ETFs |
| Question | Which stocks to buy | Which asset class to rotate into |
| Data | Sharadar direct (point-in-time, delisted included) | yfinance daily closes |
| Entry point | `opt-factor` · `opt-factor-tui` | `make run` · `run.py` |

**The two are fully isolated at the code level** — neither imports the other; the only shared symbol is `config.RISK_FREE_RATE`.

---

# 1. Factor engine

A cross-sectional US equity factor engine built so that results can be **trusted, not just produced**.
One design principle drives everything: **never fail silently.**

## Why this engine

Quant backtests fail in a small number of well-known ways. Each one is blocked structurally here.

| Common failure | How it is prevented |
|---|---|
| **Survivorship bias** — only today's survivors are in the sample | Delisted names retained (Enron, old American Airlines, Ambac verified present) |
| **Look-ahead** — using numbers before they were public | Expressions cannot touch raw tables; everything passes through `PanelContext`, which enforces `datekey` alignment |
| **Restatement contamination** — using revised figures | **First print wins** — only the number the market originally saw is stored |
| **Silent truncation** — partial data reported as success | Pagination raises `TruncatedDataError` when the expected range isn't reached |
| **Overfitting** — run hundreds of variants, report the best | **Deflated Sharpe Ratio** + **PBO** charge for the number of trials |
| **In-sample performance reporting** | Official performance is **walk-forward only**; single backtests are labelled reference-only |

## Performance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/performance-dark.png">
  <img alt="Cumulative growth over the walk-forward validation window, 2002-12 to 2026-08: large-cap 5-factor with 200-day timing reaches 36x at 15bps slippage and 27x at 50bps, against 14x for SPY buy-and-hold. Log vertical axis." src="docs/images/performance-light.png">
</picture>

*Log scale — equal slopes mean equal returns. The flat stretches in 2008, 2012 and
2022 are the 200-day overlay holding cash; those three are where the drawdown
parts company with SPY. (Chart labels are Korean; the alt text carries the reading.)*

<!-- PERFORMANCE:START -->

*Operating candidate · walk-forward validation window · 2002-12 – 2026-08 (23.6y)*

**Large-cap, 5 factors + 200-day moving-average timing overlay** (`configs/strategy_lean_timed.json`)

| Metric | Slippage 15bps | Slippage 50bps | SPY (same window) |
|---|---|---|---|
| CAGR | **16.34%** | 14.91% | 11.66% |
| Max drawdown | −24.3% | −24.3% | −55.2% |
| Volatility | 15.7% | 15.7% | 18.6% |
| Sharpe | **0.727** | 0.648 | 0.418 |
| Calmar | **0.67** | 0.61 | 0.21 |
| Deflated Sharpe (72 parameter trials) | **0.996** ✓ | 0.988 ✓ | — |
| Deflated Sharpe (**35 strategy trials**) | **0.988** ✓ | 0.969 ✓ | — |
| PBO (CSCV over 35 configurations) | **0.139** ✓ | — | — |

**This strategy is measured with the guards switched on** — $5 minimum price, $1M
minimum dollar volume, and slippage. The universe is the historical S&P 500, so
there is **no capacity limit**. Raising slippage to 50bps leaves drawdown and
volatility unchanged and costs 1.4pp of return.

The last two rows were added on 2026-08-17. Until then the Deflated Sharpe only
charged for the 72 parameter trials *inside* the walk-forward; the outer search —
"35 strategies were tried and one was picked" — went uncounted, a debt the
experiment log had recorded and left open. Re-measured across all 35 result series
on their common window (2009-12 onward, 201 months), it still clears at
**DSR 0.988 · PBO 0.139**.

<!-- PERFORMANCE:END -->

### Why the headline changed — the micro-cap strategy was retired

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/guards-dark.png">
  <img alt="Cumulative growth of the micro-cap strategy: 153x with the guards off, 0.040x with them on — a 96% loss of principal. Log vertical axis." src="docs/images/guards-light.png">
</picture>

*Same strategy, same window. The only difference is slippage, minimum price and
minimum dollar volume. The blue line was this README's headline until 2026-08-16.*

Until 2026-08-16 this space held a **micro-cap, 8-factor strategy** at CAGR 23.78%
and Sharpe 1.047. Switching on the three guards the design document calls mandatory
(slippage, $5 minimum price, $1M minimum dollar volume) collapsed it:
**Sharpe 1.047 → −0.224, max drawdown −23.7% → −99.2%.**

The cause was measured, not inferred — with the guards on, **98% of the universe
disappears.** At quarter-ends only 15–43 candidates remain, so the portfolio stops
being "the top 20 of a thousand" and becomes "everything that exists". Median daily
dollar volume of the actual holdings was about $45k, and two of them were **zero**.
Deployable capital caps out around $150k.

The same verification showed **slippage was not the problem** — even at a punishing
150bps the strategy clears at DSR 0.995. The liquidity filters are what broke it.

So the operating candidate moved to the large-cap variant. It had previously been
set aside for "lower returns" — but **it always had its guards on while the
micro-cap strategy had them off**, so the two had never been compared under the
same conditions. Under the same conditions it is DSR 0.996 against 0.002.

The full trail is in
[`docs/factor-system/07-experiment-log.md`](docs/factor-system/07-experiment-log.md)
§5.5 and §5.8 (Korean).

### Everything that was built, on one chart

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/risk-return-dark.png">
  <img alt="Risk-return scatter: max drawdown on the horizontal axis, CAGR on the vertical. The adopted large-cap strategy sits at 24% drawdown and 16% return, SPY at 42% and 12%, and the micro-cap strategy with guards on at 96% drawdown and −16% return." src="docs/images/risk-return-light.png">
</picture>

*Up and to the left is better. Blue is what was adopted, red is what was retired.
The cluster at the bottom — VAA-G4, BAA, 60/40 — are tactical allocation
configurations this repo built and **did not adopt**: they failed the gate at
PBO 0.770 ([`docs/taa/01-results.md`](docs/taa/01-results.md), Korean).*

**The windows differ.** The factor strategies are measured over 2002-12 – 2026-08,
the tactical allocation ones over 2008-07 – 2026-08 (218 months). That is why SPY
here (−41.8%) is a different number from SPY in the table above (−55.2%): the first
leg of the 2008 crash falls outside the shorter window. Sharing an axis is not the
same as sitting the same exam.

### Everything is published

The engine, all 158 factor definitions, and **the adopted parameters** are in
`configs/`. They were withheld for a day and then opened: the withheld recipe
(micro-cap) collapses once the guards are on, so it was never something that could
be run, and the large-cap strategy that can be run has no capacity limit and
therefore nothing to protect. Reasoning in
[`configs/README.md`](configs/README.md) (Korean).

### Before you believe that curve

**Structurally sound**: no look-ahead (parameters chosen inside each training window,
validation run once), no survivorship bias (delisted names are in the universe), no
restatements (first print wins), and commissions, slippage and liquidity filters all
switched on.

**Remaining limits**:

| | |
|---|---|
| **Window** | 23.6 years from 2002-12, 24 walk-forward folds. It contains three large drawdowns: 2008, 2020, 2022 |
| **Data frozen** | Stops at 2026-08-14 (subscription ended). No further updates |
| **Taxes** | Not modelled |
| **Costs** | Reported at both 15bps and 50bps. Realised spreads were never measured against the actual holdings |

**The Deflated Sharpe is the number that matters here.** It subtracts the maximum Sharpe you would expect from pure noise given how many variants were tried, leaving what is actually left over. A strategy that cannot clear 0.95 is not adopted — more than twenty candidates were rejected at this gate in this repository.

The full record is in [`docs/factor-system/07-experiment-log.md`](docs/factor-system/07-experiment-log.md) (Korean).

## Factor library — 158 factors

Factors are not written as 158 functions. A **declarative expression DSL** generates TTM / QoQ / YoY / acceleration variants automatically.

```python
from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.dsl.registry import factor

# Cash-based operating profitability (Ball, Gerakos, Linnainmaa & Nikolaev 2016)
CBOP = factor(
    "CBOP",
    (F.gp - _delta(F.receivables) - _delta(F.inventory) + _delta(F.liabilitiesc)) / F.assets,
    category="quality",
    direction=1,
    neutralize=("sector",),   # cross-sectional sector neutralisation
)
```

| Category | Count | Examples |
|---|---|---|
| quality | 55 | GP/A, ROIC, F-Score, accruals, net operating assets |
| growth | 26 | Revenue / earnings YoY & QoQ |
| price | 24 | Momentum 1/3/6/12M, 12-1, low volatility |
| value_price | 24 | P/E, P/B, P/S, P/FCF, P/GP |
| acceleration | 15 | Second derivative of growth |
| value_ev | 9 | EV/EBITDA, EV/GP |
| flow_proxy | 5 | 13F institutional change, insider net buying |

Only factors with a documented rationale are included — Novy-Marx (2013), Sloan (1996), Hirshleifer et al. (2004), Daniel & Titman (2006), Ball et al. (2016), plus replications from the Chen & Zimmermann open-source asset pricing library.

## Usage

> **The data in this repository stops at 2026-08-14.** The subscription has been
> ended, so there are no further updates. Outputs live in [`results/`](results/),
> which means the reported performance **can be checked but not re-run** — re-running
> needs the subscription below. The vendor's raw data is a paid product and cannot
> be redistributed; that is a licensing constraint, not a disclosure policy.
>
> **Data requirement.** The factor engine needs a [Sharadar](https://sharadar.com) subscription (Bundle, from $29/mo) — it is the only retail-priced source that provides point-in-time fundamentals *and* delisted coverage together. Without it the engine runs but has nothing to run on. The vendor adapter is isolated behind a neutral `Provider` protocol, so swapping in another source means rewriting one file. The VAA subsystem uses free yfinance data and needs no subscription.

```bash
# Ingest data (Sharadar subscription required)
export SHARADAR_API_KEY=...
opt-factor ingest --store us.duckdb --provider sharadar \
  --tables sf1,sep,daily,actions,sp500,tickers
# Loads the full Sharadar universe. To restrict it, pass --tickers-file with a
# file of your own (bulk TICKERS CSV, or tickers separated by newlines/commas).

# Screen factor predictive power — decile spread, IC, turnover
uv run python scripts/factor_lab.py --store us.duckdb --factors GP_A,PER,SIZE

# Official performance (walk-forward + Deflated Sharpe)
opt-factor optimize --store us.duckdb \
  --config configs/strategy.json \
  --space configs/space.json --objective calmar

# What to buy today (pass current holdings to get a trade plan)
opt-factor holdings --store us.duckdb \
  --config configs/strategy.json --current my_holdings.csv

# Operating console
opt-factor-tui --store us.duckdb --config configs/strategy.json
```

A strategy is fully declared by one JSON file. Below is the **retired micro-cap
strategy** (`configs/strategy_quantus_timed.json`), kept to show what switching the
guards off looks like. The operating candidate is `configs/strategy_lean_timed.json`.

```jsonc
{
  "factors": ["PER", "PSR", "POR", "PGPR",
              "NETINC_GROWTH_YOY", "OPINC_GROWTH_YOY",
              "GP_GROWTH_YOY", "REVENUE_GROWTH_YOY"],
  "universe": {
    "min_mcap_usd": 5000000, "max_mcap_usd": 80000000,
    "min_price_usd": 0.0,                      // ⚠ the design doc calls $5 mandatory
    "min_adv_usd": 0.0,                        // ⚠ the design doc calls $1M mandatory
    "exclude_financials": true, "exclude_distressed": true
  },
  "backtest": {
    "n_stocks": 20, "rebalance": "QE", "weighting": "equal",
    "max_weight": 0.06,
    "cost": {"commission_bps": 50, "slippage_bps": 0}   // ⚠ the default is 10
  },
  "timing_ma_days": 200,                       // market-timing overlay
  "timing_reentry_days": 5
}
```

> ⚠ The three marked lines **switch off guards the design document calls
> mandatory.** Switch them on and this strategy collapses (Sharpe 1.047 → −0.22).
> That verification is §5.5 of
> [`07-experiment-log.md`](docs/factor-system/07-experiment-log.md) (Korean).
> Do not run this config as-is — it is published to show what went wrong.

### Portfolio construction — what is built, and what survived

Implemented is not adopted. Each technique below ships with tests; the verdict column
records what the walk-forward said about it on this universe.

| Technique | Verdict |
|---|---|
| Market-timing overlay (Faber 200-day MA) | **Adopted** — drawdown −63.8% → −23.7% |
| Equal weighting | **Adopted** — beat all six optimised schemes (DeMiguel 1/N) |
| No-trade band (`hold_multiple`) | **Rejected** — turnover −23%, return −0.86pp |
| Regime-conditional factor weights | **Rejected** — 16.90% → 15.45%, too few samples per regime |
| Volatility targeting (Moreira & Muir 2017) | **Rejected** — alone it is worse than no timing at all (Sharpe 0.513 → 0.396) |
| Parameter ensembling (`--ensemble k`) | **Rejected** — highest CAGR in the table, but drawdown −23.7% → −30.6% and Calmar 0.71 → 0.60 |
| Sector cap (`max_sector_weight`) | **Performance-neutral** — difference from zero is not measurable (t = 0.77); kept as a risk control, not a return driver |
| In-training factor selection (IC / residual contribution) | **Not adopted** — both land within noise of the fixed 8-factor set (t ≈ 0.5); the fixed set wins on Deflated Sharpe and on having fewer moving parts |

The last three exist because measurement pointed at them, not because they sound sophisticated —
e.g. the sector cap was written after the live portfolio turned out to be 32% Technology,
which is a macro bet nobody chose to make.

## Validation tooling

| Tool | Question it answers |
|---|---|
| `scripts/factor_lab.py` | Does this factor predict anything? (decile spread · monotonicity · turnover) |
| `research/ic.py` | Rank IC · IC-IR · decay profile |
| `research/overfitting.py` | **Is this result luck?** — Deflated Sharpe · PBO (CSCV) |
| `research/regime.py` | In which market state does it work? (trend × volatility, 2×2) |
| `research/selection.py` | Factor selection inside the training window — the honest form of combination search |
| `optimize/walkforward.py` | Expanding/rolling windows, embargo, per-fold parameter stability |

Seven weighting schemes ship: equal · market-cap · inverse-volatility · risk parity · HRP · mean-variance · Black-Litterman.
**Empirically, equal weighting wins here** — the DeMiguel et al. (2009) 1/N result reproduced twice in this repository's tests.

---

# 2. VAA allocation

An implementation of Wouter Keller's Vigilant Asset Allocation, validated walk-forward.

### Momentum score (Keller 13612)

```
momentum = 12·R(1M) + 4·R(3M) + 2·R(6M) + 1·R(12M)
```

### Selection rule

- Pick the top-momentum asset from the **offensive universe** (`SPY`, `EFA`, `EEM`, `AGG`).
- If **any** offensive asset shows negative absolute momentum, treat it as a risk-off signal and rotate to the top of the **defensive universe** (`LQD`, `IEF`, `SHY`).
- 50% to the VAA selection, 12.5% each to the core sleeve (`SPY`, `TLT`, `GLD`, `BIL`) — configurable.

### Results

![15-year VAA comparison](backtest_comparison.png)

15 years, $10,000 initial. Dynamic VAA reaches **$22,813** (CAGR 5.7%, Sharpe 0.110). An
earlier version of this section reported ~$29k with Sharpe inflated 20.9× by an
annualization bug (monthly returns annualized with √252 instead of √12, fixed in
`b9043d7`) — the actual numbers are lower on both counts.

**Why CAGR sits at 5.7% instead of the ~17% cited in Keller's papers:** the strategy
spent **54.7% of months in defensive assets**, and one of them, `SHY`, held **44
months (24.4%)** at roughly 0.05% annualized yield — near zero — while `SPY`
buy-and-hold returned far more over the same stretch. Keller's headline figures come
from a 1970–2015 sample where defensive assets themselves returned 8–15%; this is not
a bug in the implementation, it is a regime the strategy's assumptions no longer
match. Full diagnosis: `docs/superpowers/specs/2026-08-17-taa-strategy-design.md` §0.

```bash
make run                     # interactive menu
python3 run.py --backtest    # dynamic VAA backtest
python3 run.py --optimize    # Sharpe-based weight optimisation
```

---

## Layout

```
src/opt_portfolio/
├── factor/                    # US equity factor engine
│   ├── data/                  #   vendor adapters · PIT store (DuckDB)
│   ├── dsl/                   #   expression tree · PIT context · registry
│   ├── library/               #   158 factor declarations
│   ├── universe/              #   liquidity, market-cap, sector filters
│   ├── portfolio/             #   score blending · 7 weighting schemes · shrinkage covariance
│   ├── backtest/              #   cross-sectional backtest · costs · market timing
│   ├── optimize/              #   walk-forward · grid/random/GP-EI search
│   ├── research/              #   IC · quantiles · DSR/PBO · regimes · factor selection
│   ├── holdings.py            #   today's picks · trade plan
│   └── tui.py                 #   operating console
├── strategies/                # VAA — momentum · asset selection · OU forecast (experimental)
├── analysis/                  # backtest · optimiser · risk · performance
├── core/                      # DuckDB incremental cache · positions
└── config.py                  # frozen dataclass settings
```

## Install & develop

```bash
make install        # uv sync --extra dev
make test           # pytest + coverage (292 tests)
make lint           # ruff check + format --check
make typecheck      # mypy src/
```

Dependencies are managed with **uv** (`uv.lock`). Do not use `pip install`.

## Documentation

Design documents are written in Korean and live in [`docs/factor-system/`](docs/factor-system/).

| File | Contents |
|---|---|
| `00-overview.md` | Design overview · data source rationale |
| `01-factor-spec.md` | Factor definitions |
| `02-universe-spec.md` | Universe filters |
| `04-data-contract.md` | **Store schema · PIT rules · vendor measurements · operating procedure** |
| `05-math-spec.md` | Weighting, backtest and walk-forward mathematics |
| `06-provider-review.md` | Comparison of 12 data vendors |
| `07-experiment-log.md` | **Experiment log — adopted strategy · rejection list · reproduction steps** |

## Limitations

**Factor engine**

- The **micro-cap universe** carries wide bid-ask spreads. The backtest assumes 0.5% commission and **zero slippage** — *that assumption is not data.* Sizing it: 2.7 portfolio turns a year means every 1% of round-trip slippage costs ~2.7%/yr, so plausible spreads erase 5–8%/yr. Measuring realised spreads against the actual holdings is the single most valuable open task in this repository.
- **Capacity is limited.** Switching to value weighting cuts Sharpe by 28%, which means the alpha sits in small names. The result will not survive at institutional size.
- **Factor selection is not charged to the Deflated Sharpe trial count.** Screening 124 factors and picking a handful is itself a search (`research/selection.py` exists to repay this debt).
- Taxes are not modelled.

**VAA**

- Optimised weights are in-sample; robustness must be checked separately.
- Fixed 0.1% transaction cost, yfinance daily closes, 5% risk-free assumption, single 15-year window.

> ⚠️ All backtests are historical and do not guarantee future returns.

## License

MIT

---

## ⭐ If this helped

If you found this useful, please **[⭐ Star](https://github.com/younghwan91/opt_portfolio)** the repository — it improves discoverability for others looking for the same thing.

- 🐛 Bugs & questions → [Issues](https://github.com/younghwan91/opt_portfolio/issues)
- 📈 Updates → [Follow @younghwan91](https://github.com/younghwan91)

## Related projects — open-source quant stack

Part of an open-source stack spanning Korean equities, US equities and crypto. Each repository stands on its own.

| Market | Project | What it is |
|---|---|---|
| 🇰🇷 Korean equities | **[kiwoom-rest-api](https://github.com/younghwan91/kiwoom-rest-api)** | Kiwoom Securities REST API client — full domestic-equity endpoint coverage, real-time WebSocket, sync + async (`pip install kiwoom-client`) |
| 🇰🇷 Korean equities | **[krx-fundamentals-api](https://github.com/younghwan91/krx-fundamentals-api)** | Korean corporate fundamentals REST API — financial statements, valuation, dividends, screening (DART + KRX + Naver) |
| 🇰🇷 Korean equities | **[krx-news-rest-api](https://github.com/younghwan91/krx-news-rest-api)** | Korean market news & disclosure collection API (FastAPI + Redis) |
| 🇰🇷 Korean equities | **[quant-airflow](https://github.com/younghwan91/quant-airflow)** | Airflow pipeline collecting Korean market data into TimescaleDB — delisted names included, so downstream backtests aren't survivorship-biased |
| 🇰🇷 Korean equities | **[kr-quant](https://github.com/younghwan91/kr-quant)** | KOSPI/KOSDAQ alpha research — walk-forward, random null controls, purged CV and Deflated Sharpe enforced as CI guardrails |
| 🇺🇸 US equities | **[automated-stock-trading-systems](https://github.com/younghwan91/automated-stock-trading-systems)** | Backtester for Bensdorp's seven non-correlated trading systems (educational reimplementation) |
| ₿ Crypto | **[quantbox-engine](https://github.com/younghwan91/quantbox-engine)** | Crypto futures backtest & execution engine — zero lookahead, backtest↔live parity |

## Author

**Younghwan Chae (채영환)** · [GitHub @younghwan91](https://github.com/younghwan91) · [LinkedIn](https://www.linkedin.com/in/younghwan-chae/)

The full open-source quant stack is listed on the [profile](https://github.com/younghwan91).
