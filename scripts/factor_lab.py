"""
팩터 연구소 — 후보 팩터를 10분할 테스트와 IC 로 한 번에 걸러낸다.

    uv run python scripts/factor_lab.py --store ~/data/us.duckdb
    uv run python scripts/factor_lab.py --store ~/data/us_micro.duckdb \
        --tickers-file ~/data/universe_mid.txt --factors CBOP,NOA,GP_A

승격 기준 (셋 다 만족해야 정식 채택 후보):
  ① 10분할 스프레드 t > 2      — 상·하위 분위 수익차가 우연이 아니다
  ② 단조성 ≥ 0.6               — 분위가 오를수록 수익도 오른다
  ③ 회전율 < 0.15              — 비용이 알파를 먹지 않는다

단조성이 낮은데 IC 만 높은 팩터는 극단값 몇 개가 만든 신호일 가능성이 크다.
10분할이 5분할보다 이 구분에 민감해서 기본값으로 쓴다.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

import opt_portfolio.factor.library  # noqa: F401,E402  (레지스트리 등록)
from opt_portfolio.factor.data.store import PITStore  # noqa: E402
from opt_portfolio.factor.dsl.registry import REGISTRY  # noqa: E402
from opt_portfolio.factor.research.ic import (  # noqa: E402
    forward_returns,
    rank_ic,
    summarize_ic,
)
from opt_portfolio.factor.research.quantiles import analyze_quantiles  # noqa: E402

PROMOTION = {"spread_t": 2.0, "monotonicity": 0.6, "turnover": 0.15}


def evaluate(ctx, spec, dates: pd.DatetimeIndex, fwd: pd.DataFrame, n_quantiles: int) -> dict:
    """한 팩터의 IC · 10분할 · 회전율."""
    panel = ctx.eval_daily(spec.scoring_expr()).reindex(dates, method="ffill")
    ic = summarize_ic(rank_ic(panel, fwd), horizon=21)
    q = analyze_quantiles(panel, fwd, n_quantiles=n_quantiles)
    turnover = float(panel.rank(axis=1, pct=True).diff().abs().mean().mean())
    passed = (
        abs(q.spread_t) >= PROMOTION["spread_t"]
        and q.monotonicity >= PROMOTION["monotonicity"]
        and turnover <= PROMOTION["turnover"]
    )
    return {
        "factor": spec.name,
        "category": spec.category,
        "ic": round(float(ic.mean), 4),
        "ic_ir": round(float(ic.ir), 3),
        "spread": round(float(q.spread) * 100, 3),
        "spread_t": round(float(q.spread_t), 2),
        "mono": round(float(q.monotonicity), 2),
        "turnover": round(turnover, 3),
        "coverage": round(float(panel.notna().mean().mean()), 3),
        "pass": "✓" if passed else "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--factors", default=None, help="쉼표 구분. 미지정 시 계산 가능한 전체")
    ap.add_argument("--quantiles", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    tickers = None
    if args.tickers_file:
        text = Path(args.tickers_file).read_text().split()
        tickers = sorted({t.strip().upper() for t in text if t.strip()})
        print(f"후보 유니버스: {len(tickers)}종목")

    store = PITStore(args.store)
    ctx = store.build_context(tickers=tickers)
    close = ctx.daily["close"]
    grid = pd.Series(close.index, index=close.index).resample("ME").last().dropna()
    dates = pd.DatetimeIndex(grid.to_numpy())
    fwd = forward_returns(close, horizon=args.horizon).reindex(dates)

    if args.factors:
        specs = [REGISTRY.get(n.strip()) for n in args.factors.split(",") if n.strip()]
    else:
        have = {"SF1", "SEP"}
        specs = [s for s in REGISTRY.all() if set(s.requires) <= have]
    print(f"{len(specs)}개 팩터 × {args.quantiles}분할 · 보유 {args.horizon}일\n")

    rows, skipped = [], []
    for i, spec in enumerate(specs, 1):
        try:
            rows.append(evaluate(ctx, spec, dates, fwd, args.quantiles))
        except Exception as exc:  # noqa: BLE001 — 한 팩터 실패로 연구소가 멈추지 않는다
            skipped.append((spec.name, type(exc).__name__))
        if i % 25 == 0:
            print(f"  {i}/{len(specs)} …", flush=True)

    df = pd.DataFrame(rows)
    df = df.reindex(df["spread_t"].abs().sort_values(ascending=False).index)
    if args.out:
        df.to_csv(args.out, index=False)
    print(f"\n평가 {len(df)} / 건너뜀 {len(skipped)}")
    print("승격 기준: |spread_t| ≥ 2 · 단조성 ≥ 0.6 · 회전율 ≤ 0.15\n")
    print(df.head(30).to_string(index=False))
    promoted = df[df["pass"] == "✓"]
    print(f"\n승격 후보 {len(promoted)}개: {', '.join(promoted['factor']) or '없음'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
