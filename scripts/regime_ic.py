"""레짐별 팩터 IC — 팩터가 시장 상태에 따라 다르게 작동하는가."""

import pathlib
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import opt_portfolio.factor.library  # noqa: F401,E402
from opt_portfolio.factor.data.store import PITStore  # noqa: E402
from opt_portfolio.factor.dsl.registry import REGISTRY  # noqa: E402
from opt_portfolio.factor.research.ic import forward_returns  # noqa: E402
from opt_portfolio.factor.research.regime import classify, factor_ic_by_regime  # noqa: E402

FACTORS = [
    "GP_A",
    "IT_TURNOVER",
    "REVENUE_GROWTH_YOY",
    "PFCR",
    "SIZE",
    "PER",
    "PSR",
    "POR",
    "PGPR",
    "NETINC_GROWTH_YOY",
    "OPINC_GROWTH_YOY",
    "GP_GROWTH_YOY",
    "MOM_12_1",
    "VOL_52W",
    "ACCRUAL_CF",
    "NOA",
]

store = PITStore("/home/young/data/us.duckdb")
ctx = store.build_context()
close = ctx.daily["close"]
grid = pd.Series(close.index, index=close.index).resample("ME").last().dropna()
dates = pd.DatetimeIndex(grid.to_numpy())
fwd = forward_returns(close, horizon=21).reindex(dates)

spy = close["SPY"].dropna()
regimes = classify(spy)
counts = regimes.reindex(dates, method="ffill").value_counts()
print("레짐별 관측 개월수:")
for k, v in counts.items():
    print(f"  {k:<18}{v:>4}개월")
print()

panels = {}
for name in FACTORS:
    try:
        panels[name] = ctx.eval_daily(REGISTRY.get(name).scoring_expr()).reindex(
            dates, method="ffill"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (건너뜀: {name} — {type(exc).__name__})")

table = factor_ic_by_regime(panels, fwd, regimes)
cols = [c for c in ["bull_calm", "bull_turbulent", "bear_calm", "bear_turbulent"] if c in table]
table = table[cols]
pathlib.Path("/home/young/data/regime_ic.csv").write_text(table.to_csv())
print((table * 100).round(2).to_string())
