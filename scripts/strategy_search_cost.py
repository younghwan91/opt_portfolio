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

주의 2 — **집계 주기가 판정을 뒤집는다. 그래서 주기를 근거로 고른다.**

    일별(4,176행)  n_blocks 8/10/12/16 → 0.657 / 0.524 / 0.599 / 0.544  전부 탈락
    월별(201개월)  n_blocks 8/10/12/16 → 0.314 / 0.155 / 0.294 / 0.278  전부 통과

seed 는 영향이 없다(0~5 전부 동일). 난수 잡음이 아니라 **방법 선택이 결론을
정하는** 자리다.

CSCV 원논문(Bailey·Borwein·López de Prado·Zhu 2016)이 기준을 준다 — Sharpe 를
쓴다면 *"IID Normal 가정이 성과의 여러 조각에서 유지될 수 있어야"* 한다.
네 주기를 그 기준으로 실측하면:

    주기    |VR-1| 중앙(독립성)   초과첨도 중앙(정규성)   관측
    일별         0.068  최상            16.29  최악        4,176
    주별         0.092                  13.19                835
    월별         0.109  양호             8.90  양호           198
    분기별       0.345  최악              2.79  최상            66

**둘을 동시에 만족하는 것은 월별뿐이다.** 일별은 첨도 16 으로 Sharpe 추정이
성립하지 않고, 분기별은 lag-1 자기상관 0.218 로 독립성이 깨지며 관측이 66개다.

일별이 틀리는 기전도 측정됐다: 분산비 VR(21)이 구성마다 0.78~1.45 로 다르다
(35개 중 20개가 1.2 초과). VR>1 은 일별이 위험을 과소평가한다는 뜻이므로,
일별 Sharpe 로 서로 비교하면 **각자 다른 크기로 부풀린 값을 비교**하게 된다.
증거는 VR 과 순위변동의 상관 **+0.659** 다 — 예를 들어 `quantus_ens3`(VR 1.44)는
일별 6위→월별 15위로 내려가고, 운용 후보 E안(VR 0.99)은 일별 13위→월별 4위다.

따라서 **표제값은 월별 · S=16(논문 권장)** 이다. 일별도 함께 출력하되 참고로
둔다 — 숨기면 이 판단 자체를 나중에 검증할 수 없다.

한 가지 더 남긴다: 월 경계를 하루씩 미는 21개 오프셋에서 PBO 는 0.143~0.468
이다. 전부 관문을 넘지만 **최악값이 한도에 가깝다.** 여유가 크지 않다.

2026-08-17 에 처음 적은 "PBO 0.139" 는 열여섯 조합 중 **최소값**이었고 방법
명시가 없었다. 주기 선택은 옳았으나 **숫자를 가장 유리한 것으로 골랐다.**
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

    _headline(monthly, n_trials)
    return 0


def _headline(monthly: pd.DataFrame, n_trials: int) -> None:
    """표제값 — 월별 · S=16. 근거는 모듈 docstring 주의 2.

    월 경계도 함께 흔든다. 달력 월말 하나만 재면 그 경계를 고른 것이 곧
    결론을 고른 것이 되고, 이 저장소는 이미 그 실수를 한 번 했다.
    """
    print("\n=== 표제값 — 월별 · S=16 (CSCV 논문 권장) ===")
    # C(16,8)=12,870 조합을 **전수** 돈다. 기본값 252 는 표본추출이라
    # 0.278 로 낮게 나오고, 전수로 가면 0.303 에서 수렴한다. 논문의
    # σ[f(λ)] 논증이 로짓 개수에 기대므로 여기서 잘라내지 않는다.
    pbo = probability_of_backtest_overfitting(monthly, n_blocks=16, max_splits=12870).pbo
    print(f"  PBO = {pbo:.3f}  {'통과' if pbo < 0.5 else '탈락'} (한도 0.5)")
    for label, name in VARIANTS.items():
        if name in monthly.columns:
            dsr = deflated_sharpe_ratio(monthly[name], n_trials=n_trials)
            print(f"  DSR({label}, n_trials={n_trials}) = {dsr:.3f}  {'✓' if dsr >= 0.95 else '✗'}")
    print(
        "\n  주기 선택 근거: Sharpe 의 IID-Normal 전제를 독립성·정규성 양쪽에서"
        "\n  만족하는 유일한 주기가 월별이다 (일별 초과첨도 16.3, 분기별 lag-1 0.218)."
        "\n  일별 결과도 위에 남긴다 — 숨기면 이 판단을 나중에 검증할 수 없다."
    )


if __name__ == "__main__":
    raise SystemExit(main())
