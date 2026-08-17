# opt_portfolio

**English** · [한국어](README.ko.md)

**US equity factor engine + tactical asset allocation (TAA) validation system.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

Three subsystems live in this repository.

| | **Factor engine** (`factor/`) | **TAA allocation** (`taa/`) | **Original VAA** (`strategies/`) |
|---|---|---|---|
| Scope | US single stocks (20,931 tickers, 1997–2026) | 18 ETFs | 7–11 ETFs |
| Question | Which stocks to buy | Which asset class to rotate into | (same — first attempt) |
| Data | Sharadar direct (point-in-time, delisted included) | Sharadar funds bulk (`closeadj`) | yfinance daily closes |
| Entry point | `opt-factor` · `opt-factor-tui` | `scripts/run_taa.py` | `make run` · `run.py` |
| Outcome | **1 adopted** (large-cap) | **0 adopted** — all 9 failed the PBO gate | Kept as the record of why it failed |

**There is one isolation rule and one exception.** None of the three imports another.
The single exception is `taa/` → `factor.research.overfitting` (DSR and PBO): those
functions take a plain return series, so the coupling is thin, and **not trusting
performance produced without a gate is this repository's standing rule.** Imports in
the other direction are forbidden.

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
| Deflated Sharpe (**35 strategy trials**) | **0.982** ✓ | 0.957 ✓ | — |
| PBO (CSCV over 35 configs · monthly · S=16) | **0.303** ✓ | — | — |

**This strategy is measured with the guards switched on** — $5 minimum price, $1M
minimum dollar volume, and slippage. The universe is the historical S&P 500, so
there is **no capacity limit**. Raising slippage to 50bps leaves drawdown and
volatility unchanged and costs 1.4pp of return.

The last two rows charge for the search **outside** the walk-forward — "35 strategies
were tried and one was picked". The reasoning behind the gate's settings (monthly,
S=16) is below. Reproduce with `uv run python scripts/strategy_search_cost.py`; it
reads only `results/oos/` and needs no vendor data.

<!-- PERFORMANCE:END -->

### Which ruler measures the gate — corrected twice

PBO **flips its verdict with the aggregation frequency and the block count.**
Measured here:

```
Daily (4,176 rows)   S=8/10/12/16 → 0.657 / 0.524 / 0.599 / 0.544   all fail
Monthly (201 months) S=8/10/12/16 → 0.314 / 0.155 / 0.294 / 0.278   all pass
```

Varying the seed from 0 to 5 changes nothing to three decimal places — **this is not
sampling noise, it is a place where the choice of method decides the answer.** So the
choice has to be argued, not assumed.

The CSCV paper (Bailey, Borwein, López de Prado & Zhu, 2016) supplies the criterion:
if the metric is the Sharpe ratio, *"the IID Normal distribution assumption [must] be
maintained on various slices of the reported performance"*, and **S = 16 is a
reasonable value in most cases**. Measured against that criterion:

| Frequency | Obs | Independence (VR−1, abs) | Normality (excess kurtosis) | PBO (S=16) |
|---|---|---|---|---|
| Daily | 4,176 | **0.068** best | **16.29** worst | 0.544 ✗ |
| Weekly | 835 | 0.092 | 13.19 | 0.538 ✗ |
| **Monthly** | **198** | **0.109** good | **8.90** good | **0.303** ✓ |
| Quarterly | 66 | 0.345 worst | 2.79 best | 0.675 ✗ |

**Monthly is the only frequency that satisfies both.** Daily has excess kurtosis of
16, so the Sharpe estimator itself is not valid there; quarterly has the best
normality but lag-1 autocorrelation of 0.218 and only 66 observations.

The mechanism behind daily's failure was measured, not assumed. **The variance ratio
VR(21) ranges 0.78–1.45 across configurations** (20 of 35 exceed 1.2). VR > 1 means
daily returns understate that configuration's risk, so ranking 35 configs by daily
Sharpe compares **numbers inflated by different amounts**. The evidence is a +0.659
correlation between VR and rank change: `quantus_ens3` (VR 1.44) drops from 6th daily
to 15th monthly, while the operating candidate (VR 0.99) rises from 13th to **4th**.

#### This spot was wrong twice

1. **2026-08-17, morning** — published "DSR 0.988 · PBO **0.139**". That was the
   **minimum** of sixteen combinations, with no method stated and no script to
   reproduce it.
2. **The same evening** — finding it did not reproduce, the claim was withdrawn
   entirely as *"does not clear the gate"*. **That was also wrong** — the number was
   wrong, not the conclusion, and it was withdrawn without measuring which frequency
   was correct.

> The first error ran in the flattering direction and the second in the unflattering
> one. **Different directions, same cause: one option picked without a reason.**

**What remains.** Across the 21 possible month-boundary offsets, PBO ranges
0.143–0.468. All clear the gate, but **the worst case sits close to 0.5.** And the 35
configurations span different universes, whereas CSCV assumes parameter variants of a
single strategy. There is only one way to remove that limitation: **pre-register one
configuration and test it as a single hypothesis.**

```bash
uv run python scripts/strategy_search_cost.py   # reproduces the table above
```

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
> **Data requirement.** The factor engine needs a [Sharadar](https://sharadar.com) subscription (Bundle, from $29/mo) — it is the only retail-priced source that provides point-in-time fundamentals *and* delisted coverage together. Without it the engine runs but has nothing to run on. The vendor adapter is isolated behind a neutral `Provider` protocol, so swapping in another source means rewriting one file. `taa/` uses the same Sharadar bulk (the funds tables). The only thing that runs without a subscription is the original VAA in `strategies/` (yfinance).

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
| Market-timing overlay (Faber 200-day MA) | **Adopted** — drawdown −63.8% → −23.7% (measured on the micro-cap universe; the operating large-cap candidate uses the same overlay) |
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

# 2. Tactical asset allocation (TAA)

Rotating between a handful of ETFs each month rather than picking single stocks.
There is exactly one result here — **nine configurations were pre-registered and
all nine were rejected.** How that happened is this section.

## It started with VAA, and VAA failed

Wouter Keller's Vigilant Asset Allocation (2017) was the starting point.

```
momentum = 12·R(1M) + 4·R(3M) + 2·R(6M) + 1·R(12M)      ← Keller 13612W
```

- Buy the top-momentum asset from the **offensive universe** (`SPY`, `EFA`, `EEM`, `AGG`).
- But if **any** of the four shows negative absolute momentum, treat it as risk-off and
  rotate to the top of the **defensive universe** (`LQD`, `IEF`, `SHY`) — the breadth rule.

The papers report 16–17% a year. **Measured here it is 6.07%** (2008-07 to 2026-08,
218 months). The implementation is not wrong; the cause measures out like this:

| What | Measured |
|---|---|
| Share of months spent in **defensive** assets | **55.7%** |
| Of which, months parked in `SHY` alone | 44 — yielding ~0.05% at the time, effectively cash |
| Keller's validation window | 1970–2015 — **when defensive assets themselves paid 8–15%** |

So VAA spends more than half its life somewhere "safe", and **since 2008 the safe
place pays nothing.** The premise does not match the era. Full diagnosis in the
[design document](docs/superpowers/specs/2026-08-17-taa-strategy-design.md) §0 (Korean).

> Separately, an earlier version of this README inflated the Sharpe by **20.9×** —
> monthly returns annualized with √252 instead of √12 (fixed in `b9043d7`). Returns
> were right and only the risk metrics were wrong, which is why it survived so long.
> Regression test: `tests/test_risk_annualization.py`.

## Keller fixed it twice himself

VAA's real problem is that **the four offensive assets are both the investment
universe and the alarm.** Through 2011–2026 `EEM` and `EFA` were chronically weak, so
**assets there was never any intention of buying pushed the whole portfolio
defensive.** `SPY` rose throughout.

This diagnosis is not ours — Keller acknowledged the same problem and published
successors. **That the lineage exists is independent confirmation of the diagnosis.**

| Year | Strategy | What changed | Here |
|---|---|---|---|
| 2017 | **VAA** | (the original problem) | implemented as a baseline |
| 2018 | DAA | **canary universe** — separates the alarm from the investment universe | **not implemented** — the rule source 404'd, and an unverified spec does not get implemented (CLAUDE.md §4) |
| 2022 | **BAA** | canary + separate offensive set + wider defensive set + different selection metric | **six variants centred here** |
| 2023 | HAA | built for inflation and rising rates | **not implemented — and no reason was recorded.** The design document listed it in this table and then walked past it |

That last row is this round's loose end. What was skipped is recorded as skipped.

## So a replacement was built — and rejected too

BAA separates the canary (alarm) assets from the investment universe. Centred on it,
nine configurations were **registered before any result was seen** and measured in
one pass.

| Configuration | CAGR | Max DD | Calmar | DSR |
|---|---|---|---|---|
| `spy` (baseline) | 12.46% | −41.8% | 0.298 | 0.968 |
| `static_60_40` (baseline) | 8.87% | −25.1% | 0.354 | 0.986 |
| `vaa_g4` | 6.07% | −20.9% | 0.290 | 0.871 |
| `baa_agg` | 8.82% | −16.5% | 0.535 | 0.976 |
| `baa_bal` | 7.28% | −11.1% | 0.654 | 0.994 |
| `baa_agg_ma` | 8.81% | −13.4% | 0.656 | 0.985 |
| `baa_bal_tranche` | 8.11% | −11.3% | 0.717 | 0.996 |
| `baa_bal_ma_tranche` | 7.32% | −10.0% | 0.731 | 0.997 |
| `baa_bal_ma` | 6.72% | **−8.3%** | **0.812** | 0.997 |

**PBO = 0.770 → nothing adopted.** The gate was not relaxed.

PBO 0.77 means *"pick the in-sample winner among these nine and it lands below median
out-of-sample 77% of the time."* It does **not** mean nothing works — and that
distinction is the most important thing in this section. All six BAA variants beat
60/40 on Calmar, without exception (0.535–0.812 against 0.354). At the same time,
PBO across just those six is **0.861**: which one is best cannot be determined from
this data.

> Both sentences have to stand together. "BAA beats 60/40" and "which BAA is best is
> unknown" are **both true**, and collapsing to either one alone gives a wrong answer.

Testing this honestly requires **picking one BAA configuration in advance and
pre-registering it against 60/40 as a single hypothesis.** Choosing the
best-performing variant after the fact and declaring it the winner is a mistake this
repository already made once on the factor side.

## Defects caught in this round

**With nothing adopted, these are worth more than the results.** Almost every one
pushed performance in the flattering direction — and mistakes in that direction give
you no reason to suspect them, so they live longest.

This is the complete list from the source document — §3 (one methodological defect)
and §4 (six implementation defects) — plus one recorded only in code comments.

| # | Defect | Effect |
|---|---|---|
| §3 | Tranches **shifted the whole price panel**, so each sleeve measured a different period | Smoothing, not diversification. Sleeve correlation 0.381 (0.819 once fixed). Fixing it moved **PBO 0.139 → 0.770** |
| §4-1 | Return labels were off by one month (a defect in the plan, not the code) | Labelling by realisation month, not decision month, is what keeps it look-ahead free |
| §4-2 | The spec said the common window began 2007-06; it actually begins 2008-07 (`BIL` listing + warm-up) | 230 months → **218 months** |
| §4-3 | Configuration 9 was a **silent duplicate** of configuration 8 (an `if`/`elif` dispatch) | Only 8 of the 9 pre-registered configs actually differed |
| §4-4 | §5 made PBO the primary gate but the §6 adoption formula omitted it | The passive baselines (`spy`, `60/40`) get rejected for "PBO exceeded" — though they are not products of the search |
| §4-5 | `run_with_ma_overlay` and `run_with_tranches` ignored the "prepend principal to equity" convention | Tranche max DD −9.35% against an actual −10.99% — and the error favoured **the four improvement candidates** specifically |
| §4-6 | The same missing-principal bug existed **separately** in `summarize()` | `baa_bal_tranche` max DD −9.72% → −11.31%. Fixing one does not fix the other |
| (code) | `pandas.pct_change()` pads missing values by default, **fabricating 0% returns** | `fill_method=None` forced across every signal (`taa/signals.py`) |

The full trail, including the pre-registered predictions checked against the results,
is in [`docs/taa/01-results.md`](docs/taa/01-results.md) (Korean).

```bash
uv run python scripts/run_taa.py    # 9 configurations · PBO · verdict table

make run                            # original VAA (yfinance, kept for the record)
python3 run.py --backtest
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
├── taa/                       # tactical allocation — 9 pre-registered configs · PBO gate
│   ├── data.py                #   Sharadar funds bulk → dividend-adjusted price panel
│   ├── signals.py             #   13612W momentum · 200-day moving average
│   ├── strategy.py            #   StrategySpec (canary / offensive / defensive universes)
│   ├── backtest.py            #   monthly rebalance · MA overlay · tranches
│   ├── registry.py            #   the 9 pre-registered configurations · N_TRIALS
│   └── evaluate.py            #   DSR · PBO · adoption verdict
├── strategies/                # original VAA — momentum · asset selection · OU forecast (experimental)
├── analysis/                  # backtest · optimiser · risk · performance
├── core/                      # DuckDB incremental cache · positions
└── config.py                  # frozen dataclass settings
```

## Install & develop

```bash
make install        # uv sync --extra dev
make test           # pytest + coverage (399 tests)
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
| [`taa/01-results.md`](docs/taa/01-results.md) | **TAA results — 9 configurations, 0 adopted, 6 defects caught** |

## Limitations

**Factor engine (operating candidate = the large-cap strategy)**

- **The window contains only three major drawdowns** (2008, 2020, 2022). A strategy whose claim is drawdown defence rests that claim on three events. This is the largest remaining limitation.
- **Transaction costs are an assumption, not a measurement.** Both 15bps and 50bps are reported and the conclusion holds at either (DSR 0.996 / 0.988), but **the realised bid-ask spread of the actual holdings has never been measured.** The universe is the historical S&P 500, so this is less dangerous than it was for micro caps.
- **Data stops at 2026-08-14** (subscription ended). No further updates — `ingest` now fails, and that is not a bug.
- Taxes are not modelled.

> **One debt repaid.** This spot used to read *"factor selection is not charged to the
> Deflated Sharpe trial count."* On 2026-08-17 all 35 result series were re-measured on
> their common window and cleared the gate at **DSR 0.988 · PBO 0.139** — the last two
> rows of the performance table above.

**TAA**

- **Nothing was adopted, so there is no operating conclusion here.** PBO 0.770 was not relaxed.
- 218 months of sample, containing the same three drawdowns.
- The final month is **half a month** — the data stops on 2026-08-14, so `to_monthly` labels half a month's move as a full one. `run_taa.py` prints a warning.
- Nothing before an ETF's listing is visible. The papers extend to the 1970s using index proxies; only real ETF prices were used here.

**Original VAA (`strategies/`, kept for the record)**

- Optimised weights are in-sample. Fixed 0.1% transaction cost, yfinance daily closes, single 15-year window.
- **These numbers differ from `vaa_g4` in `taa/`** — different data source, different window. For comparison, use the `taa/` figures.

> ⚠️ All backtests are historical and do not guarantee future returns.

## License

MIT

---

## ⭐ If this helped

If you found this useful, please **[⭐ Star](https://github.com/younghwan91/portfolio-research)** the repository — it improves discoverability for others looking for the same thing.

- 🐛 Bugs & questions → [Issues](https://github.com/younghwan91/portfolio-research/issues)
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
