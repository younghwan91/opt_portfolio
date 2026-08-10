"""
파라미터 탐색 — 그리드 / 랜덤 / 베이지안 (GP-EI)

퀀트 관점:
- 탐색 알고리즘보다 중요한 것은 **모든 시도의 기록**이다. n_trials 를
  모르면 Deflated Sharpe 를 계산할 수 없고, 시도별 수익률 행렬이 없으면
  PBO 를 계산할 수 없다. 그래서 탐색기는 결과와 함께 전체 시도 로그를
  반드시 반환한다 — 최고값만 돌려주는 인터페이스는 의도적으로 없다.
- 베이지안(GP-EI)은 평가 1회가 비싼 walk-forward 안에서 그리드보다
  적은 시도로 같은 품질에 도달하기 위한 것이지, '더 좋은 최적값'을
  찾기 위한 것이 아니다. 시도가 적을수록 SR₀ 가 낮아져 DSR 이 유리해진다
  — 효율적 탐색은 통계적 정직성과 같은 방향이다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from scipy import stats

# 파라미터 공간 정의:
#   {"n_stocks": ("int", 10, 50), "ir_scale": ("float", 0.01, 0.10),
#    "weighting": ("cat", ["equal", "hrp", "black_litterman"])}
ParamSpec = tuple
ParamSpace = dict[str, ParamSpec]
Params = dict[str, Any]


@dataclass
class Trial:
    params: Params
    objective: float


@dataclass
class SearchResult:
    best_params: Params
    best_objective: float
    trials: list[Trial] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def objectives(self) -> np.ndarray:
        return np.asarray([t.objective for t in self.trials])


# ------------------------------------------------------------------ 인코딩


def _encode(params: Params, space: ParamSpace) -> np.ndarray:
    """파라미터 → [0,1]^d. GP 커널 거리 계산용."""
    x = []
    for name, spec in space.items():
        kind, *rest = spec
        v = params[name]
        if kind in ("float", "int"):
            lo, hi = rest
            x.append((v - lo) / (hi - lo) if hi > lo else 0.5)
        elif kind == "cat":
            choices = rest[0]
            # 범주는 인덱스 매핑 — 거리 의미는 없지만 GP 가 같은 값끼리
            # 묶는 데는 충분하다 (엄밀한 처리는 원-핫이지만 차원 낭비)
            x.append(choices.index(v) / max(len(choices) - 1, 1))
        else:
            raise ValueError(f"알 수 없는 파라미터 유형: {kind}")
    return np.asarray(x)


def sample_params(space: ParamSpace, rng: np.random.Generator) -> Params:
    out: Params = {}
    for name, spec in space.items():
        kind, *rest = spec
        if kind == "float":
            lo, hi = rest
            out[name] = float(rng.uniform(lo, hi))
        elif kind == "int":
            lo, hi = rest
            out[name] = int(rng.integers(lo, hi + 1))
        elif kind == "cat":
            out[name] = rest[0][int(rng.integers(len(rest[0])))]
        else:
            raise ValueError(f"알 수 없는 파라미터 유형: {kind}")
    return out


def grid_params(space: ParamSpace, steps: int = 4) -> list[Params]:
    """각 축을 steps 개로 이산화한 전체 격자."""
    axes: list[list[Any]] = []
    for spec in space.values():
        kind, *rest = spec
        if kind == "float":
            lo, hi = rest
            axes.append(list(np.linspace(lo, hi, steps)))
        elif kind == "int":
            lo, hi = rest
            vals = np.unique(np.linspace(lo, hi, steps).round().astype(int))
            axes.append(list(vals))
        else:
            axes.append(list(rest[0]))
    names = list(space.keys())
    return [dict(zip(names, combo, strict=True)) for combo in product(*axes)]


# ------------------------------------------------------------------ 탐색기


def search(
    evaluate: Callable[[Params], float],
    space: ParamSpace,
    *,
    method: str = "bayesian",
    n_trials: int = 32,
    grid_steps: int = 4,
    seed: int = 0,
) -> SearchResult:
    """
    목적함수를 최대화하는 파라미터 탐색.

    Args:
        evaluate: params → 목적값 (클수록 좋음). 실패 시 예외를 던지면
            해당 시도는 -inf 로 기록되고 탐색은 계속된다.
        method: "grid" | "random" | "bayesian"
    """
    rng = np.random.default_rng(seed)

    if method == "grid":
        candidates = grid_params(space, grid_steps)
        return _evaluate_all(evaluate, candidates)
    if method == "random":
        candidates = [sample_params(space, rng) for _ in range(n_trials)]
        return _evaluate_all(evaluate, candidates)
    if method == "bayesian":
        return _bayesian(evaluate, space, n_trials, rng)
    raise ValueError(f"알 수 없는 탐색 방법: {method}")


def _evaluate_all(evaluate: Callable[[Params], float], candidates: list[Params]) -> SearchResult:
    trials = [Trial(p, _safe_eval(evaluate, p)) for p in candidates]
    best = max(trials, key=lambda t: t.objective)
    return SearchResult(best.params, best.objective, trials)


def _safe_eval(evaluate: Callable[[Params], float], params: Params) -> float:
    try:
        v = float(evaluate(params))
        return v if np.isfinite(v) else -np.inf
    except Exception:
        return -np.inf


def _bayesian(
    evaluate: Callable[[Params], float],
    space: ParamSpace,
    n_trials: int,
    rng: np.random.Generator,
) -> SearchResult:
    """
    GP + Expected Improvement.

    - 초기 max(4, n/4) 회는 랜덤 (GP 워밍업)
    - 커널: RBF, 길이척도 0.25 (정규화 공간 기준), 관측 노이즈 1e-4
    - 획득함수 최적화: 256개 랜덤 후보 중 EI 최대 선택
      (연속 최적화보다 단순하지만 범주형 혼합 공간에서 더 강건)
    """
    n_init = max(4, n_trials // 4)
    trials: list[Trial] = []

    for i in range(n_trials):
        if i < n_init or not _has_finite(trials):
            params = sample_params(space, rng)
        else:
            params = _propose_ei(trials, space, rng)
        trials.append(Trial(params, _safe_eval(evaluate, params)))

    best = max(trials, key=lambda t: t.objective)
    return SearchResult(best.params, best.objective, trials)


def _has_finite(trials: list[Trial]) -> bool:
    return any(np.isfinite(t.objective) for t in trials)


def _propose_ei(trials: list[Trial], space: ParamSpace, rng: np.random.Generator) -> Params:
    done = [t for t in trials if np.isfinite(t.objective)]
    x_obs = np.stack([_encode(t.params, space) for t in done])
    y_obs = np.asarray([t.objective for t in done])

    y_mean, y_std = y_obs.mean(), y_obs.std()
    y_norm = (y_obs - y_mean) / y_std if y_std > 0 else y_obs * 0.0

    length_scale, noise = 0.25, 1e-4
    k_xx = _rbf(x_obs, x_obs, length_scale) + noise * np.eye(len(x_obs))
    k_inv_y = np.linalg.solve(k_xx, y_norm)
    k_inv = np.linalg.inv(k_xx)

    candidates = [sample_params(space, rng) for _ in range(256)]
    x_cand = np.stack([_encode(p, space) for p in candidates])
    k_star = _rbf(x_cand, x_obs, length_scale)

    mu = k_star @ k_inv_y
    var = np.clip(1.0 - np.einsum("ij,jk,ik->i", k_star, k_inv, k_star), 1e-12, None)
    sigma = np.sqrt(var)

    best_y = y_norm.max()
    z = (mu - best_y) / sigma
    ei = sigma * (z * stats.norm.cdf(z) + stats.norm.pdf(z))
    return candidates[int(np.argmax(ei))]


def _rbf(a: np.ndarray, b: np.ndarray, length_scale: float) -> np.ndarray:
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return np.asarray(np.exp(-0.5 * d2 / length_scale**2))
