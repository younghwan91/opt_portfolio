"""
README 성과 표·그래프 갱신 — 저장된 OOS 수익률에서 자동 생성한다.

    uv run python scripts/update_readme_performance.py \
        --oos ~/data/oos_quantus_timed.json --benchmark-store ~/data/us_micro.duckdb

왜 자동화하는가: 성과 숫자를 손으로 옮기면 반드시 어긋난다. 이 저장소는
이미 그런 종류의 사고를 여러 번 겪었다 — 문서와 코드가 갈리면 문서를
믿을 수 없게 되고, 믿을 수 없는 문서는 없느니만 못하다.

**종목명은 절대 쓰지 않는다.** 초소형주 유니버스라 공개 추천이 몰리면
자신의 체결가가 나빠지고, 벤더 라이선스 문제도 별개로 남는다.
공개하는 것은 성과와 방법이지 보유 목록이 아니다.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
START = "<!-- PERFORMANCE:START -->"
END = "<!-- PERFORMANCE:END -->"


def metrics(returns: pd.Series, ann: int = 252) -> dict[str, float]:
    from opt_portfolio.factor.optimize.walkforward import (
        annualized_calmar,
        annualized_sharpe,
    )

    r = returns.fillna(0.0)
    equity = (1.0 + r).cumprod()
    years = len(r) / ann
    return {
        "cagr": float(equity.iloc[-1]) ** (1 / years) - 1,
        "mdd": float((equity / equity.cummax() - 1.0).min()),
        "vol": float(r.std(ddof=1) * np.sqrt(ann)),
        "sharpe": annualized_sharpe(r),
        "calmar": annualized_calmar(r),
        "years": years,
    }


def benchmark_returns(store_path: str, index: pd.DatetimeIndex, ticker: str) -> pd.Series | None:
    from opt_portfolio.factor.data.store import PITStore

    with PITStore(store_path) as store:
        frame = store.conn.execute(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date", [ticker]
        ).fetch_df()
    if frame.empty:
        return None
    close = frame.set_index(pd.to_datetime(frame["date"]))["close"]
    return close.pct_change().reindex(index).fillna(0.0)


def sparkline(equity: pd.Series, width: int = 60) -> str:
    """의존성 없이 그리는 누적수익 스파크라인 — 로그 축."""
    blocks = "▁▂▃▄▅▆▇█"
    sampled = equity.iloc[:: max(1, len(equity) // width)]
    logs = np.log(sampled.to_numpy())
    lo, hi = logs.min(), logs.max()
    if hi <= lo:
        return blocks[0] * len(logs)
    scaled = ((logs - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in scaled)


def build_block(
    oos: pd.Series, bench: pd.Series | None, label: str, trials: int | None = None
) -> str:
    m = metrics(oos)
    rows = [
        "| Metric | Strategy | " + ("SPY (same window) |" if bench is not None else ""),
        "|---|---|" + ("---|" if bench is not None else ""),
    ]

    def row(name: str, key: str, fmt: str) -> str:
        cell = format(m[key], fmt)
        if bench is None:
            return f"| {name} | **{cell}** |"
        b = format(metrics(bench)[key], fmt)
        return f"| {name} | **{cell}** | {b} |"

    rows += [
        row("CAGR", "cagr", ".2%"),
        row("Max drawdown", "mdd", ".1%"),
        row("Volatility", "vol", ".1%"),
        row("Sharpe", "sharpe", ".3f"),
        row("Calmar", "calmar", ".2f"),
    ]
    if trials:
        from opt_portfolio.factor.research.overfitting import deflated_sharpe_ratio

        dsr = deflated_sharpe_ratio(oos, trials)
        cell = f"**{dsr:.3f}**" + (" ✓" if dsr >= 0.95 else "")
        rows.append(f"| **Deflated Sharpe** ({trials} trials) | {cell} |" + (" — |" if bench is not None else ""))

    equity = (1.0 + oos.fillna(0.0)).cumprod()
    period = f"{oos.index.min():%Y-%m} – {oos.index.max():%Y-%m}"
    lines = [
        START,
        "",
        f"*{label} · walk-forward out-of-sample · {period} ({m['years']:.1f}y)*",
        "",
        *rows,
        "",
        f"```\n{sparkline(equity)}\n```",
        "",
        f"Cumulative {equity.iloc[-1]:.1f}× over the validation window. "
        "Holdings are not published — the universe is micro-cap and crowding "
        "would move the entry price.",
        "",
        END,
    ]
    return "\n".join(lines)


def splice(path: Path, block: str) -> bool:
    text = path.read_text()
    if START not in text or END not in text:
        return False
    updated = re.sub(
        re.escape(START) + r".*?" + re.escape(END), block.replace("\\", "\\\\"), text, flags=re.S
    )
    if updated == text:
        return False
    path.write_text(updated)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oos", required=True, help="walk-forward OOS 수익률 JSON")
    ap.add_argument("--benchmark-store", default=None)
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--label", default="Adopted strategy")
    ap.add_argument("--trials", type=int, default=None, help="DSR 정산에 쓸 총 시도 횟수")
    ap.add_argument("--readme", action="append", default=None)
    args = ap.parse_args(argv)

    oos = pd.read_json(args.oos, typ="series").sort_index()
    bench = (
        benchmark_returns(args.benchmark_store, oos.index, args.benchmark)
        if args.benchmark_store
        else None
    )
    block = build_block(oos, bench, args.label, args.trials)

    targets = [Path(p) for p in (args.readme or ["README.md"])]
    for path in targets:
        full = path if path.is_absolute() else REPO / path
        if not full.exists():
            print(f"건너뜀 (없음): {full}")
            continue
        print(("갱신: " if splice(full, block) else "표식 없음/변화 없음: ") + str(full))
    print()
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
