"""9개 구성 평가와 채택 판정.

**PBO 가 주 관문이다.** 여기서의 탐색은 파라미터 적합이 아니라 *9개 구성 중
하나 고르기* 이고, CSCV 가 정확히 그 상황을 다룬다 — 인샘플 1등이 아웃샘플에서
중앙값 아래로 떨어질 확률.

DSR 은 보조다. **월별 수익률을 그대로 넘긴다 — 연율화 금지** (docstring 요구).
이 저장소는 연율화 주기를 이미 두 번 틀렸다.

**아홉 개가 같은 구간을 보지 않으면 비교 자체가 성립하지 않는다.** BAA 계열은
BIL 상장(2007-05) 이후 12개월 모멘텀 워밍업이 끝나야 시작하지만 spy·
static_60_40·vaa_g4 는 더 일찍 시작한다 — 실측으로 230개월 대 219개월 차이가
난다. 절단 없이 지표를 계산하면 "60/40 을 이겼다"는 채택 기준이 BAA 에게는
2008년 폭락 11개월을 뺀 구간에서, 60/40 에게는 포함한 구간에서 매겨지는
꼴이 된다. `evaluate_all` 은 아홉 개 수익률 인덱스의 교집합으로 **먼저** 자른
뒤에만 지표를 계산한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..factor.research.overfitting import deflated_sharpe_ratio
from .backtest import run_backtest, run_with_ma_overlay, run_with_tranches
from .registry import MA_OVERLAY, N_TRIALS, REGISTERED, TRANCHE

#: 설계 문서 §6 의 채택 기준. 결과를 보고 바꾸지 않는다.
ADOPTION: dict[str, float] = {
    "mdd_limit": -0.20,
    "dsr_gate": 0.95,
    "pbo_limit": 0.5,
}

_MONTHS_PER_YEAR = 12


def summarize(name: str, returns: pd.Series) -> dict[str, float | str]:
    """월별 수익률 → 지표. 연율화는 12 로 한다.

    MDD 는 원금(진입 시점, 값 1.0)을 equity 곡선 맨 앞에 붙인 뒤 잰다. 안
    붙이면 첫 달 손실이 자기 자신의 cummax 가 되어 그 낙폭이 통째로 빠진다
    — `backtest.py` 의 `_equity_from_returns` 가 같은 이유로 원금을 붙이는
    것과 같은 규약이다. `BacktestOutput.equity` 는 그 규약을 이미 지키지만
    이 함수는 공통 구간으로 자른 `returns`(`evaluate_all` 의 `matrix`)만
    받으므로 여기서 다시 붙여야 한다. 실측: `baa_bal_tranche` 는 이 누락으로
    MDD 가 −9.72%로 보고됐으나 실제는 −11.33%다 (Calmar 0.835 → 0.716).
    """
    equity = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()], ignore_index=True)
    years = len(returns) / _MONTHS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    mdd = float((equity / equity.cummax() - 1).min())
    vol = float(returns.std() * np.sqrt(_MONTHS_PER_YEAR))
    return {
        "name": name,
        "cagr": cagr,
        "mdd": mdd,
        "vol": vol,
        "calmar": cagr / abs(mdd) if mdd else float("nan"),
        "sharpe": float(returns.mean() * _MONTHS_PER_YEAR / vol) if vol else float("nan"),
        # DSR 은 기간(월별) 수익률을 그대로 받는다 — 연율화한 값을 넘기면 공식의
        # 왜도/첨도 보정이 기간 단위 SR 을 전제한다는 가정이 깨진다.
        "dsr": float(deflated_sharpe_ratio(returns, n_trials=N_TRIALS)),
        "months": float(len(returns)),
    }


def common_window(returns: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """모든 구성 수익률 인덱스의 교집합.

    지표를 계산하기 **전에** 이걸로 자른다. 그렇지 않으면 늦게 시작하는
    구성(BAA 계열)과 일찍 시작하는 구성(spy·60/40·vaa_g4)이 서로 다른 길이의
    구간에서 비교되고, 그중 하나가 2008년 폭락처럼 결과를 지배하는 구간을
    빼먹은 채 "이겼다"고 판정될 수 있다.
    """
    if not returns:
        raise ValueError("공통 구간을 계산할 수익률이 없다")
    idx: pd.DatetimeIndex | None = None
    for r in returns.values():
        idx = r.index if idx is None else idx.intersection(r.index)
    assert idx is not None
    return idx.sort_values()


def evaluate_all(
    daily: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """등록된 9개 구성을 모두 돌리고, 공통 구간으로 자른 뒤 지표를 계산한다.

    Returns:
        (지표표, 수익률 행렬) — 행렬은 공통 구간 × 9개 구성, PBO 입력용.
    """
    rets: dict[str, pd.Series] = {}
    defensive: dict[str, float] = {}
    for name, spec in REGISTERED.items():
        kw = {"start": start, "end": end, "cost_bps": cost_bps}
        # 순서가 중요하다: baa_bal_ma_tranche 는 TRANCHE 와 MA_OVERLAY 양쪽에
        # 다 들어 있다. TRANCHE 를 먼저 검사하고 ma_overlay 플래그를 같이
        # 넘겨야 두 오버레이가 동시에 적용된다 — elif 로 떨어뜨리면 트랜치만
        # 적용되어 config 8(baa_bal_tranche)의 조용한 중복이 된다.
        if name in TRANCHE:
            out = run_with_tranches(spec, daily, ma_overlay=name in MA_OVERLAY, **kw)  # type: ignore[arg-type]
        elif name in MA_OVERLAY:
            out = run_with_ma_overlay(spec, daily, **kw)  # type: ignore[arg-type]
        else:
            out = run_backtest(spec, daily, **kw)  # type: ignore[arg-type]
        rets[name] = out.returns
        defensive[name] = out.defensive_ratio

    window = common_window(rets)
    if len(window) == 0:
        raise ValueError("9개 구성의 공통 구간이 비었다 — 데이터 구간을 확인하라")

    matrix = pd.DataFrame({name: r.reindex(window) for name, r in rets.items()})
    if matrix.isna().any().any():
        # window 는 정의상 모든 인덱스의 교집합이므로 이 자리에 결측이 남으면
        # reindex 대상이 아니라 원본 시리즈 자체에 구멍(조용한 절단)이 있다는
        # 뜻이다 — 여기서 잡아내야지 지표 계산에 흘려보내면 안 된다.
        raise ValueError("공통 구간인데 결측이 남았다 — 원본 수익률 시리즈를 의심하라")

    rows = [{**summarize(name, matrix[name]), "defensive": defensive[name]} for name in REGISTERED]
    metrics = pd.DataFrame(rows).set_index("name")
    return metrics, matrix


def verdict(
    metrics: pd.DataFrame,
    pbo: float,
    baseline_calmar: float,
    baseline_name: str | None = None,
) -> pd.DataFrame:
    """채택 기준 적용. 하나라도 못 넘으면 기각하고 이유를 적는다.

    `baseline_name` 을 넘기면 그 행(보통 60/40)에는 Calmar-vs-베이스라인
    관문을 적용하지 않는다. 베이스라인은 후보가 비교당하는 **기준선**이지
    그 자신과 비교될 후보가 아니다 — `calmar <= baseline_calmar` 는
    베이스라인 자신에게 늘 참이라(자기 자신과 "초과"가 성립할 수 없다)
    이 예외가 없으면 static_60_40 은 다른 성적과 무관하게 항상 기각된다.
    지정하지 않으면(`None`) 전 행에 동일하게 적용된다 — 기존 동작 그대로다.
    """
    rows = []
    for name, row in metrics.iterrows():
        reasons: list[str] = []
        if pbo > ADOPTION["pbo_limit"]:
            reasons.append(f"PBO {pbo:.2f} > {ADOPTION['pbo_limit']}")
        if row["mdd"] < ADOPTION["mdd_limit"]:
            reasons.append(f"MDD {row['mdd']:.1%} 가 한도 초과")
        if row["dsr"] < ADOPTION["dsr_gate"]:
            reasons.append(f"DSR {row['dsr']:.3f} < {ADOPTION['dsr_gate']}")
        if name != baseline_name and row["calmar"] <= baseline_calmar:
            reasons.append(f"Calmar {row['calmar']:.2f} 가 60/40({baseline_calmar:.2f}) 이하")
        rows.append({"name": name, "adopted": not reasons, "reason": " · ".join(reasons) or "—"})
    return pd.DataFrame(rows).set_index("name")
