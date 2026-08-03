"""
공분산 추정 — Ledoit-Wolf 상수상관 수축

퀀트 관점:
- 종목 20~100개 × 관측 252일이면 표본 공분산은 조건수가 나빠
  MVO 가 추정오차를 '최적화'해버린다 (error maximization).
- Ledoit-Wolf (2004) 는 표본 공분산을 상수상관 타깃으로 수축하는
  최적 강도 δ* 를 닫힌형으로 준다. 하이퍼파라미터가 없어 PO 대상이
  아니라는 점이 중요하다 — 추정기까지 튜닝하면 과적합 표면적만 넓어진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf_cc(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    상수상관 타깃 Ledoit-Wolf 수축 공분산.

    Args:
        returns: (기간 × 자산) 수익률. NaN 포함 행은 제거된다 —
            상장 기간이 다른 종목이 섞이면 호출 측에서 공통 구간을 잘라줄 것.

    Returns:
        (수축 공분산, 수축 강도 δ ∈ [0,1]).
        δ=0 이면 표본 그대로, δ=1 이면 타깃 그대로.
    """
    clean = returns.dropna(how="any")
    t_obs, n = clean.shape
    if t_obs < 2 or n < 2:
        raise ValueError(f"공분산 추정에 관측 부족: {t_obs}기간 × {n}자산")

    x = clean.to_numpy(dtype=float)
    x = x - x.mean(axis=0)

    sample = x.T @ x / t_obs
    var = np.diag(sample).copy()
    sd = np.sqrt(var)

    corr = sample / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # π: 표본 공분산 원소들의 점근 분산 합
    y = x**2
    pi_mat = y.T @ y / t_obs - sample**2
    pi_hat = pi_mat.sum()

    # ρ: 타깃과 표본의 점근 공분산 (대각 + 비대각 상관 항)
    theta_mat = (x**3).T @ x / t_obs - var[:, None] * sample
    np.fill_diagonal(theta_mat, 0.0)
    rho_hat = np.diag(pi_mat).sum() + rbar * (np.outer(1.0 / sd, sd) * theta_mat).sum()

    # γ: 타깃과 표본의 거리
    gamma_hat = np.linalg.norm(sample - target, "fro") ** 2

    if gamma_hat <= 0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa / t_obs, 0.0, 1.0))

    shrunk = delta * target + (1.0 - delta) * sample
    cov = pd.DataFrame(shrunk, index=clean.columns, columns=clean.columns)
    return cov, delta


def annualize(cov_daily: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    return cov_daily * periods


def nearest_psd(cov: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    """음의 고유값을 잘라 PSD 를 보장한다 — 수치 오차 방어용."""
    vals, vecs = np.linalg.eigh(cov.to_numpy())
    vals = np.clip(vals, eps, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(fixed, index=cov.index, columns=cov.columns)
