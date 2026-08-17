"""
README 성과 표·그래프 갱신 — 저장된 OOS 수익률에서 자동 생성한다.

    uv run python scripts/update_readme_performance.py \
        --oos ~/data/oos_quantus_timed.json --benchmark-store ~/data/us_micro.duckdb

왜 자동화하는가: 성과 숫자를 손으로 옮기면 반드시 어긋난다. 이 저장소는
이미 그런 종류의 사고를 여러 번 겪었다 — 문서와 코드가 갈리면 문서를
믿을 수 없게 되고, 믿을 수 없는 문서는 없느니만 못하다.

**현재 이 스크립트는 실행하면 README 를 퇴행시킨다.** 마커 사이의 표를
2026-08-17 에 손으로 늘렸다 — 슬리피지 50bps 열, 전략 탐색 35회 DSR, PBO 행.
아래 `build_block()` 은 여전히 단일 열 표를 만들고, 이미 철회된 문장("보유
종목은 공개하지 않는다")을 각주로 다시 쓴다. 되살리려면 `build_block()` 을
현재 표 형태에 맞춰 고친 뒤 쓰라. 그 전까지는 표를 손으로 관리한다.

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


# 차트는 이 스크립트가 만들지 않는다 — `scripts/make_readme_charts.py` 가
# `docs/images/performance-*.png` 의 단일 소유자다. 한때 두 스크립트가 같은
# 파일명에 서로 다른 그림을 써서, 나중에 실행한 쪽이 조용히 이겼다.
# 여기는 마커 사이의 **표**만 갱신하고, 그림은 마커 바깥에 둔다.


#: 언어별 표 문구. 한국어 README 에 영문 표가 들어가 있으면 자동 생성이라는
#: 사실만 드러나고 읽는 사람에게는 불친절하다.
STRINGS = {
    "en": {
        "metric": "Metric",
        "strategy": "Strategy",
        "bench": "SPY (same window)",
        "period": "*{label} · walk-forward out-of-sample · {period} ({years:.1f}y)*",
        "rows": ["CAGR", "Max drawdown", "Volatility", "Sharpe", "Calmar"],
        "dsr": "**Deflated Sharpe** ({trials} trials)",
        "footer": (
            "Cumulative {mult:.1f}× over the validation window. Holdings are not "
            "published — the universe is micro-cap and crowding would move the entry price."
        ),
    },
    "ko": {
        "metric": "지표",
        "strategy": "전략",
        "bench": "SPY (같은 구간)",
        "period": "*{label} · walk-forward 검증 구간 · {period} ({years:.1f}년)*",
        "rows": ["연평균 수익률", "최대낙폭", "변동성", "Sharpe", "Calmar"],
        "dsr": "**Deflated Sharpe** (시도 {trials}회)",
        "footer": (
            "검증 구간 누적 {mult:.1f}배. **보유 종목은 공개하지 않는다** — 초소형주라 "
            "공개 추천이 몰리면 자신의 체결가가 나빠진다."
        ),
    },
}


def build_block(
    oos: pd.Series,
    bench: pd.Series | None,
    label: str,
    trials: int | None = None,
    lang: str = "en",
    note: str | None = None,
) -> str:
    s = STRINGS[lang]
    m = metrics(oos)
    rows = [
        f"| {s['metric']} | {s['strategy']} | " + (f"{s['bench']} |" if bench is not None else ""),
        "|---|---|" + ("---|" if bench is not None else ""),
    ]

    def row(name: str, key: str, fmt: str) -> str:
        cell = format(m[key], fmt)
        if bench is None:
            return f"| {name} | **{cell}** |"
        b = format(metrics(bench)[key], fmt)
        return f"| {name} | **{cell}** | {b} |"

    keys = ["cagr", "mdd", "vol", "sharpe", "calmar"]
    fmts = [".2%", ".1%", ".1%", ".3f", ".2f"]
    rows += [row(n, k, f) for n, k, f in zip(s["rows"], keys, fmts)]
    if trials:
        from opt_portfolio.factor.research.overfitting import deflated_sharpe_ratio

        dsr = deflated_sharpe_ratio(oos, trials)
        cell = f"**{dsr:.3f}**" + (" ✓" if dsr >= 0.95 else "")
        name = s["dsr"].format(trials=trials)
        rows.append(f"| {name} | {cell} |" + (" — |" if bench is not None else ""))

    equity = (1.0 + oos.fillna(0.0)).cumprod()
    period = f"{oos.index.min():%Y-%m} – {oos.index.max():%Y-%m}"
    lines = [
        START,
        "",
        s["period"].format(label=label, period=period, years=m["years"]),
        "",
        *rows,
        "",
        s["footer"].format(mult=equity.iloc[-1]),
    ]
    # 구간을 늘려 표제 숫자가 커졌다면 그 사실을 표 옆에 함께 둔다. 각주로
    # 밀어두면 읽는 사람은 큰 숫자만 가져간다.
    if note:
        lines += ["", note]
    lines += ["", END]
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
    ap.add_argument("--label", default=None, help="기본값은 언어별 문구")
    ap.add_argument("--trials", type=int, default=None, help="DSR 정산에 쓸 총 시도 횟수")
    ap.add_argument(
        "--note",
        action="append",
        default=None,
        help="표 아래에 붙일 단서. `lang:문장` 형식 (예: ko:2003~2007 은 ...)",
    )
    ap.add_argument(
        "--readme",
        action="append",
        default=None,
        help="README 경로. `경로:lang` 형식으로 언어 지정 (예: README.ko.md:ko)",
    )
    args = ap.parse_args(argv)

    oos = pd.read_json(args.oos, typ="series").sort_index()
    bench = (
        benchmark_returns(args.benchmark_store, oos.index, args.benchmark)
        if args.benchmark_store
        else None
    )

    default_label = {"en": "Adopted strategy", "ko": "채택 전략"}
    for target in args.readme or ["README.md"]:
        name, _, lang = target.partition(":")
        lang = lang or "en"
        full = Path(name) if Path(name).is_absolute() else REPO / name
        if not full.exists():
            print(f"건너뜀 (없음): {full}")
            continue
        notes = dict(n.split(":", 1) for n in (args.note or []))
        block = build_block(
            oos, bench, args.label or default_label[lang], args.trials, lang, notes.get(lang)
        )
        print(("갱신: " if splice(full, block) else "표식 없음/변화 없음: ") + str(full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
