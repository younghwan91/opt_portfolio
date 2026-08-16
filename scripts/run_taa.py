"""사전 등록 9개 구성을 돌리고 PBO/DSR 로 판정한다.

    uv run python scripts/run_taa.py

결과가 나쁘면 목록을 늘리고 싶어진다. **늘리지 않는다** — 그 순간 DSR 이
의미를 잃는다.
"""

from __future__ import annotations

import pandas as pd

from opt_portfolio.factor.research.overfitting import probability_of_backtest_overfitting
from opt_portfolio.taa.data import load_prices
from opt_portfolio.taa.evaluate import evaluate_all, verdict
from opt_portfolio.taa.registry import REGISTERED

START, END = pd.Timestamp("2007-06-30"), pd.Timestamp("2026-08-31")
COST_BPS = 10.0
_MIN_MONTHS = 200


def main() -> int:
    tickers = sorted({t for spec in REGISTERED.values() for t in spec.tickers()})
    daily = load_prices(tickers)
    print(
        f"가격 패널: {daily.shape[1]}종목 {daily.index.min().date()} ~ {daily.index.max().date()}"
    )

    metrics, matrix = evaluate_all(daily, start=START, end=END, cost_bps=COST_BPS)

    # 아홉 구성이 전부 같은 구간에서 비교됐는지를 보여준다 — 개별 months 가
    # 200 미만이면 데이터 구간이 의심스러운 상태이므로 여기서 멈춘다.
    start_d, end_d = matrix.index.min().date(), matrix.index.max().date()
    print(f"\n공통 구간: {start_d} ~ {end_d} ({len(matrix)}개월)")
    too_short = metrics.index[metrics["months"] < _MIN_MONTHS]
    if len(too_short) > 0:
        raise SystemExit(
            f"다음 구성이 {_MIN_MONTHS}개월 미만이다 — 데이터 구간을 확인하라: {list(too_short)}"
        )

    pbo = probability_of_backtest_overfitting(matrix).pbo

    print("\n=== 지표 ===")
    print(metrics.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nPBO = {pbo:.3f}  (관측 {len(matrix)}개월 × 구성 {matrix.shape[1]}개)")
    print("\n=== 판정 ===")
    print(verdict(metrics, pbo, float(metrics.loc["static_60_40", "calmar"])).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
