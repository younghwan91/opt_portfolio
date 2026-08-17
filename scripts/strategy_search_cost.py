#!/usr/bin/env python3
"""
전략 탐색 자체의 비용을 정산한다 — walk-forward **바깥**의 시도 횟수.

`optimize` 가 보고하는 DSR 은 walk-forward **안쪽**의 파라미터 시도만 센다.
그런데 이 저장소는 그 위에서 한 겹 더 골랐다 — 35개 전략 구성을 돌리고
하나를 운용 후보로 정했다. **그 고르는 행위 자체가 탐색이다.**

이 부채는 `07-experiment-log.md` 에 오래 적혀 있었고 2026-08-17 에 갚았다.
그런데 그때는 일회성으로 재고 숫자만 README 에 적었다 — 재현 스크립트가
없어 나중에 대조할 수 없었다. 이 파일이 그 구멍을 메운다.

    uv run python scripts/strategy_search_cost.py

`results/oos/` 의 35개 산출물만 쓴다. 벤더 데이터가 필요 없으므로
**구독 없이도 이 숫자는 재현된다.**

주의 1: 35개는 서로 다른 구간을 덮는다(학습 구간 설정이 다르다). PBO 는
모든 구성이 같은 관측 격자 위에 있어야 하므로 **공통 구간으로 자른다** —
자르고 남은 길이를 반드시 출력한다. 조용히 줄어들면 그것이 이 저장소의
지배적 실패 유형이다.

주의 2 — **이 스크립트가 존재하는 진짜 이유.** PBO 는 집계 주기와 블록 수에
따라 판정이 **뒤집힌다**:

    일별(4,176행)  n_blocks 8/10/12/16 → 0.657 / 0.524 / 0.599 / 0.544  전부 탈락
    월별(201개월)  n_blocks 8/10/12/16 → 0.314 / 0.155 / 0.294 / 0.278  전부 통과

seed 는 영향이 없다(0~5 전부 동일). 즉 이것은 난수 잡음이 아니라 **방법 선택이
결론을 정하는** 자리다. 2026-08-17 에는 월별·n_blocks=10 하나만 재서 README 에
"PBO 0.139 ✓" 라고 적었다 — 유리한 쪽이었고, 유리한 쪽이라 의심받지 않았다.

그래서 이 스크립트는 **양쪽을 다 출력하고 어느 쪽도 대표로 고르지 않는다.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from opt_portfolio.factor.research.overfitting import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

REPO = Path(__file__).resolve().parents[1]
OOS = REPO / "results" / "oos"

#: 운용 후보와 그 슬리피지 민감도. DSR 은 "고른 것" 하나에 대해 잰다.
ADOPTED = "oos_lean_timed_train5"
VARIANTS = {"15bps": "oos_lean_timed_train5", "50bps": "oos_lean_timed_slip50"}


def load(path: Path) -> pd.Series:
    raw = json.loads(path.read_text())
    idx = pd.DatetimeIndex([pd.Timestamp(int(k), unit="s") for k in raw])
    return pd.Series(list(raw.values()), index=idx, dtype=float).sort_index()


def main() -> int:
    paths = sorted(OOS.glob("oos_*.json"))
    if not paths:
        raise SystemExit(f"산출물이 없다: {OOS}")

    series = {p.stem: load(p) for p in paths}
    n_trials = len(series)

    frame = pd.DataFrame(series)
    total_rows = len(frame)
    common = frame.dropna(how="any")
    if common.empty:
        raise SystemExit("공통 구간이 비었다 — 구성들이 겹치는 날짜가 없다")

    print(f"구성 {n_trials}개 · 합집합 {total_rows}행 → 공통 구간 {len(common)}행")
    print(f"공통 구간: {common.index[0]:%Y-%m-%d} ~ {common.index[-1]:%Y-%m-%d}")
    dropped = total_rows - len(common)
    # 조용히 줄이지 않는다 — 얼마나 잘렸는지 항상 말한다.
    print(f"공통 구간으로 자르며 버린 행: {dropped} ({dropped / total_rows:.1%})")

    if ADOPTED not in common.columns:
        raise SystemExit(f"채택 전략 산출물이 없다: {ADOPTED}")

    monthly = (1.0 + common).resample("ME").prod() - 1.0
    print(f"월별 집계: {len(monthly)}개월")

    for freq, frame_at in (("일별", common), ("월별", monthly)):
        print(f"\n--- {freq} ---")
        # 블록 수를 하나만 쓰지 않는다. 하나만 쓰면 그 하나를 고른 것이
        # 곧 결론을 고른 것이 된다.
        for n_blocks in (8, 10, 12, 16):
            pbo = probability_of_backtest_overfitting(frame_at, n_blocks=n_blocks).pbo
            mark = "통과" if pbo < 0.5 else "탈락"
            print(f"  PBO(n_blocks={n_blocks:2d}) = {pbo:.3f}  {mark}")
        for label, name in VARIANTS.items():
            if name not in frame_at.columns:
                print(f"  DSR({label}): 산출물 없음 ({name})")
                continue
            dsr = deflated_sharpe_ratio(frame_at[name], n_trials=n_trials)
            print(f"  DSR({label}, n_trials={n_trials}) = {dsr:.3f}  {'✓' if dsr >= 0.95 else '✗'}")

    print(
        "\n판정: 일별과 월별이 서로 다른 답을 준다. 한쪽만 인용하지 마시라 —"
        "\n      어느 쪽을 고르느냐가 관문 통과 여부를 정한다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
