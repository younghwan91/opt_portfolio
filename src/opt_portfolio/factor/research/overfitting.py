"""
과최적화 정산 — Deflated Sharpe Ratio, PBO (CSCV)

이 모듈이 이 레포의 '양심'이다. 136개 팩터 × 파라미터 공간을 탐색하면
순수한 노이즈에서도 Sharpe 2 가 나온다. 여기의 두 도구가 그걸 정산한다.

- **DSR** (Bailey & López de Prado 2014): N 번 시도했을 때 노이즈만으로
  기대되는 최대 Sharpe(SR₀)를 빼고, 수익률의 왜도·첨도까지 반영해
  "진짜 알파일 확률"을 돌려준다.
- **PBO** (Bailey et al. 2015, CSCV): 표본을 블록으로 쪼개 조합적으로
  IS/OOS 를 뒤집으며, IS 최고 전략이 OOS 에서 중앙값 미만으로 떨어지는
  빈도를 잰다. PBO > 0.5 면 선택 과정 자체가 노이즈 피팅이라는 뜻이다.

퀀트 관점:
- 파라미터 최적화(PO)는 반드시 이 모듈을 통과한 결과만 신뢰한다.
  optimize/ 레이어는 모든 시도를 기록해 n_trials 를 여기로 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

#: Euler–Mascheroni 상수 — 최대값 기댓값 근사에 사용
_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """
    N 번의 독립 시도에서 노이즈만으로 기대되는 최대 Sharpe (기간 단위).

        SR₀ = √V[SR] · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]

    Args:
        n_trials: 시도 횟수 (탐색한 파라미터 조합 수)
        sr_var: 시도들 간 Sharpe 의 분산 (기간 단위)
    """
    if n_trials <= 1 or sr_var <= 0:
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(sr_var) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sr_var_across_trials: float | None = None,
) -> float:
    """
    선택된 전략이 '진짜'일 확률 (0..1).

    Args:
        returns: 선택된 전략의 기간 수익률 (일별이면 일별 그대로 — 연율화 금지.
            공식의 왜도/첨도 보정이 기간 단위 SR 을 전제한다)
        n_trials: 이 전략을 고르기까지 탐색한 총 조합 수
        sr_var_across_trials: 시도들 간 SR 분산. 미지정 시 보수적으로
            선택 전략 SR 분산의 추정치(1/T)를 사용

    Returns:
        DSR ∈ [0,1]. 0.95 이상이면 다중검정을 감안해도 유의.
    """
    r = returns.dropna()
    t_obs = len(r)
    if t_obs < 10 or r.std(ddof=1) == 0:
        return np.nan

    sr = float(r.mean() / r.std(ddof=1))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    if sr_var_across_trials is None:
        sr_var_across_trials = (1.0 + 0.5 * sr**2) / t_obs

    sr0 = expected_max_sharpe(n_trials, sr_var_across_trials)
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    z = (sr - sr0) * np.sqrt(t_obs - 1) / denom
    return float(stats.norm.cdf(z))


# ------------------------------------------------------------------ PBO (CSCV)


@dataclass(frozen=True)
class PBOResult:
    pbo: float                 # OOS 순위가 중앙값 미만인 빈도
    logits: np.ndarray         # 조합별 λ = ln(ω/(1−ω))
    n_splits: int
    oos_sharpe_of_is_best: np.ndarray  # IS 최적 전략의 OOS Sharpe 분포

    @property
    def is_overfit(self) -> bool:
        return self.pbo > 0.5


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    n_blocks: int = 10,
    max_splits: int = 252,
    seed: int = 0,
) -> PBOResult:
    """
    CSCV 로 백테스트 과적합 확률을 추정한다.

    Args:
        trial_returns: (기간 × 전략구성) — 탐색한 각 파라미터 조합의
            수익률 시계열. optimize/ 레이어가 자동으로 쌓아준다.
        n_blocks: 시계열을 자를 블록 수 (짝수). 블록 단위로 자르는 이유는
            일별 셔플이 자기상관을 파괴해 PBO 를 과소평가하기 때문.
        max_splits: 평가할 IS/OOS 조합 수 상한 (C(10,5)=252 는 전수)

    구현 노트: 각 조합에서 IS Sharpe 최댓값 전략을 뽑고, 그 전략의
    OOS 상대순위 ω 를 구한다. ω < 0.5 (중앙값 미만) 빈도가 PBO.
    """
    m = trial_returns.dropna(how="all").fillna(0.0).to_numpy()
    t_obs, n_cfg = m.shape
    if n_cfg < 2 or t_obs < n_blocks * 2:
        raise ValueError(
            f"PBO 에는 전략 2개 이상, 관측치 {n_blocks * 2}개 이상이 필요합니다 "
            f"(현재 {n_cfg}개 전략 × {t_obs}개 관측)"
        )

    blocks = np.array_split(np.arange(t_obs), n_blocks)
    all_combos = list(combinations(range(n_blocks), n_blocks // 2))
    if len(all_combos) > max_splits:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_combos), size=max_splits, replace=False)
        all_combos = [all_combos[i] for i in idx]

    def _sharpe(block: np.ndarray) -> np.ndarray:
        mu = block.mean(axis=0)
        sd = block.std(axis=0, ddof=1)
        return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)

    logits, oos_sr_best = [], []
    for combo in all_combos:
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate(
            [blocks[b] for b in range(n_blocks) if b not in combo]
        )
        best = int(np.argmax(_sharpe(m[is_rows])))
        oos_sr = _sharpe(m[oos_rows])
        # OOS 상대순위 ω ∈ (0,1): 1 이면 OOS 에서도 1등
        omega = (stats.rankdata(oos_sr)[best]) / (n_cfg + 1.0)
        logits.append(np.log(omega / (1.0 - omega)))
        oos_sr_best.append(oos_sr[best])

    logits_arr = np.asarray(logits)
    return PBOResult(
        pbo=float((logits_arr < 0).mean()),
        logits=logits_arr,
        n_splits=len(all_combos),
        oos_sharpe_of_is_best=np.asarray(oos_sr_best),
    )
