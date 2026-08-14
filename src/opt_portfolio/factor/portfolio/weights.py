"""
비중 결정 스킴 6종 — equal / inverse_vol / risk_parity / hrp / mvo / black_litterman

퀀트 관점 (스킴 선택의 수학적 근거):
- **equal**: 기대수익·공분산 추정을 전혀 신뢰하지 않을 때의 미니맥스 선택.
  DeMiguel et al. (2009) 이후 '1/N 을 이기기'가 최적화의 실질 기준선이다.
- **inverse_vol**: 상관을 무시한 리스크 패리티 근사. 추정 대상이 분산뿐이라
  강건하지만, 상관 구조가 뚜렷하면 (예: 같은 섹터 몰림) 위험이 집중된다.
- **risk_parity**: 각 종목의 위험기여도(RC)를 균등화. Spinu (2013) 의
  볼록 정식화 min ½wᵀΣw − Σbᵢln wᵢ 를 풀면 유일해가 보장된다.
- **hrp** (López de Prado 2016): 공분산 역행렬을 아예 쓰지 않는다.
  상관 → 거리 → 계층 클러스터링 → 재귀 이분할. 추정오차에 가장 강건.
- **mvo**: 평균-분산. 기대수익 μ 가 필요한데, 팩터 스코어는 기대수익이
  아니다 — 그래서 μ = z-score × ir_scale 로 명시적으로 변환하고,
  수축 공분산 + 비중 상한으로 극단해를 막는다.
- **black_litterman**: 시장균형 π = δΣw_mkt 를 사전분포로, 팩터 스코어를
  view 로 주입. 멀티팩터 스코어는 '상대적 견해'이지 수익률 예측치가
  아니므로, BL 이 이론적으로 가장 정합한 접합점이다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize as sp_opt

from opt_portfolio.factor.portfolio.covariance import ledoit_wolf_cc, nearest_psd

#: z-score 1 단위당 연간 기대 초과수익 — MVO/BL 의 μ 변환 계수.
#: PO 대상 파라미터 (실증 문헌의 팩터 프리미엄 스케일 2~5% 에서 출발).
DEFAULT_IR_SCALE = 0.03

#: BL 사전분포 불확실성 τ — 관례적 0.01~0.05 구간
DEFAULT_TAU = 0.05

#: 시장 위험회피계수 δ — (E[r_m]−rf)/σ²_m ≈ 2.5 (미장 장기 평균)
DEFAULT_RISK_AVERSION = 2.5


# ------------------------------------------------------------------ 공통 유틸


def cap_and_normalize(w: pd.Series, max_weight: float) -> pd.Series:
    """
    비중 상한을 지키며 합계 1 로 정규화 (water-filling).

    상한에 걸린 종목의 초과분을 나머지에 비례 배분하고, 그로 인해 새로
    상한을 넘는 종목이 생기면 반복한다. 전 종목이 상한에 걸리면
    (n × max_weight < 1) 균등 분배로 폴백.
    """
    if len(w) * max_weight < 1.0 - 1e-9:
        return pd.Series(1.0 / len(w), index=w.index)

    w = w.clip(lower=0.0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(w), index=w.index)

    for _ in range(64):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def _prep_cov(returns: pd.DataFrame) -> pd.DataFrame:
    cov, _ = ledoit_wolf_cc(returns)
    return nearest_psd(cov)


# ------------------------------------------------------------------ 스킴 구현


def equal_weight(names: pd.Index) -> pd.Series:
    return pd.Series(1.0 / len(names), index=names)


def value_weight(names: pd.Index, market_caps: pd.Series | None) -> pd.Series:
    """
    시총가중 — Hou·Xue·Zhang(2020)이 아노말리 검증의 표준으로 요구하는 방식.

    균등가중은 소형주에 사실상 레버리지를 거는 것과 같아 호가 스프레드 반동으로
    수익이 부풀려진다. HXZ 는 이 하나를 바꾸는 것만으로 452개 아노말리의 65%가
    유의성을 잃는다고 보고한다. '균등가중에서만 사는 결과'를 걸러내는 장치다.

    시총이 없으면(초기 히스토리) 균등가중으로 후퇴한다 — 조용히 0 을 주면
    그 종목이 포트폴리오에서 사라진다.
    """
    if market_caps is None:
        return equal_weight(names)
    caps = pd.to_numeric(market_caps.reindex(names), errors="coerce")
    caps = caps.where(caps > 0)
    if caps.notna().sum() == 0:
        return equal_weight(names)
    # 시총 결측 종목은 관측된 것들의 중앙값으로 채운다 (탈락시키지 않는다)
    caps = caps.fillna(caps.median())
    return caps / caps.sum()


def inverse_vol(returns: pd.DataFrame) -> pd.Series:
    vol = returns.std(ddof=1)
    inv = 1.0 / vol.where(vol > 0)
    inv = inv.fillna(inv.mean())
    return inv / inv.sum()


def risk_parity(returns: pd.DataFrame) -> pd.Series:
    """
    Spinu (2013): min ½wᵀΣw − Σ bᵢ ln wᵢ  (bᵢ = 1/n)

    볼록이라 유일해. ∇=Σw − b/w 가 0 이 되는 점에서
    wᵢ(Σw)ᵢ = bᵢ·상수, 즉 위험기여도 균등이 성립한다.
    """
    cov = _prep_cov(returns).to_numpy()
    n = cov.shape[0]
    b = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        return float(0.5 * w @ cov @ w - b @ np.log(w))

    def grad(w: np.ndarray) -> np.ndarray:
        return np.asarray(cov @ w - b / w)

    res = sp_opt.minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        jac=grad,
        method="L-BFGS-B",
        bounds=[(1e-9, None)] * n,
    )
    w = res.x / res.x.sum()
    return pd.Series(w, index=returns.columns)


def hrp(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (López de Prado 2016).

    ① 상관 → 거리 d = √((1−ρ)/2) → 단일연결 클러스터링
    ② 덴드로그램 잎 순서로 공분산을 준대각화
    ③ 재귀 이분할: 클러스터 분산(역분산 가중)이 작은 쪽에 더 배분
    """
    from scipy.cluster import hierarchy as sch
    from scipy.spatial.distance import squareform

    cov = _prep_cov(returns)
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    dist = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    link = sch.linkage(squareform(dist, checks=False), method="single")
    order = sch.leaves_list(link)
    names = returns.columns[order]

    cov_np = cov.loc[names, names].to_numpy()
    w = np.ones(len(names))
    clusters: list[np.ndarray] = [np.arange(len(names))]

    def _cluster_var(idx: np.ndarray) -> float:
        sub = cov_np[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        half = len(cluster) // 2
        left, right = cluster[:half], cluster[half:]
        var_l, var_r = _cluster_var(left), _cluster_var(right)
        alpha = 1.0 - var_l / (var_l + var_r)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        clusters += [left, right]

    return pd.Series(w / w.sum(), index=names).reindex(returns.columns)


def mvo(
    returns: pd.DataFrame,
    scores: pd.Series,
    *,
    max_weight: float = 0.10,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    ir_scale: float = DEFAULT_IR_SCALE,
) -> pd.Series:
    """
    평균-분산 최적화:  max μᵀw − (δ/2)·wᵀΣw,  Σw=1, 0 ≤ w ≤ cap

    μ 는 팩터 z-score 를 ir_scale 로 연수익률 단위 변환한 것.
    """
    mu = _scores_to_mu(scores, returns.columns, ir_scale)
    cov = _prep_cov(returns).to_numpy() * 252.0
    return _solve_mvo(mu, cov, returns.columns, max_weight, risk_aversion)


def black_litterman(
    returns: pd.DataFrame,
    scores: pd.Series,
    *,
    market_caps: pd.Series | None = None,
    max_weight: float = 0.10,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    tau: float = DEFAULT_TAU,
    ir_scale: float = DEFAULT_IR_SCALE,
    view_confidence: float = 0.5,
) -> pd.Series:
    """
    Black-Litterman: 시장균형 사전분포 + 팩터 view 의 베이지안 결합.

        π = δ·Σ·w_mkt                       (균형 기대수익)
        Q = π + ir_scale·z                  (팩터 견해, P = I)
        Ω = diag(τΣ)/confidence             (견해 불확실성)
        μ_post = π + τΣ·(τΣ+Ω)⁻¹·(Q−π)

    confidence → 0 이면 μ_post → π (시장 포트폴리오로 수렴),
    confidence → ∞ 면 μ_post → Q (순수 팩터 베팅). 이 하나의 파라미터가
    '팩터를 얼마나 믿는가'를 연속적으로 조절한다 — PO 의 핵심 축.
    """
    names = returns.columns
    cov = _prep_cov(returns).to_numpy() * 252.0

    if market_caps is not None:
        w_mkt = market_caps.reindex(names).fillna(0.0)
        w_mkt = (
            (w_mkt / w_mkt.sum()).to_numpy()
            if w_mkt.sum() > 0
            else np.full(len(names), 1.0 / len(names))
        )
    else:
        w_mkt = np.full(len(names), 1.0 / len(names))

    pi = risk_aversion * cov @ w_mkt
    z = _scores_to_mu(scores, names, 1.0)  # 정규화된 z
    q = pi + ir_scale * z

    tau_cov = tau * cov
    omega = np.diag(np.diag(tau_cov)) / max(view_confidence, 1e-6)
    mu_post = pi + tau_cov @ np.linalg.solve(tau_cov + omega, q - pi)

    return _solve_mvo(mu_post, cov, names, max_weight, risk_aversion)


# ------------------------------------------------------------------ 내부


def _scores_to_mu(scores: pd.Series, names: pd.Index, scale: float) -> np.ndarray:
    """스코어를 횡단면 z 로 표준화 후 수익률 단위로 스케일."""
    s = scores.reindex(names).astype(float)
    z = (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
    return np.asarray((z.fillna(0.0) * scale).to_numpy())


def _solve_mvo(
    mu: np.ndarray,
    cov: np.ndarray,
    names: pd.Index,
    max_weight: float,
    risk_aversion: float,
) -> pd.Series:
    n = len(names)
    cap = max(max_weight, 1.0 / n)  # cap 이 1/n 미만이면 실행 불가능

    def objective(w: np.ndarray) -> float:
        return float(-(mu @ w) + 0.5 * risk_aversion * w @ cov @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return np.asarray(-mu + risk_aversion * cov @ w)

    res = sp_opt.minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = pd.Series(np.clip(res.x, 0.0, None), index=names)
    return cap_and_normalize(w, cap)


# ------------------------------------------------------------------ 디스패치

WeightFn = Callable[..., pd.Series]

SCHEMES = frozenset(
    {"equal", "value", "inverse_vol", "risk_parity", "hrp", "mvo", "black_litterman"}
)


def compute_weights(
    scheme: str,
    returns: pd.DataFrame,
    scores: pd.Series,
    *,
    max_weight: float = 0.10,
    market_caps: pd.Series | None = None,
    **kwargs: float,
) -> pd.Series:
    """
    스킴 이름 → 비중. 백테스트 엔진과 PO 레이어가 쓰는 단일 진입점.

    scheme 자체가 범주형 PO 파라미터다 — walk-forward 안에서
    'equal 을 이기는 스킴이 실제로 있는가'를 데이터가 답하게 한다.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"알 수 없는 비중 스킴 '{scheme}'. 지원: {sorted(SCHEMES)}")

    if scheme == "equal":
        w = equal_weight(returns.columns)
    elif scheme == "value":
        w = value_weight(returns.columns, market_caps)
    elif scheme == "inverse_vol":
        w = inverse_vol(returns)
    elif scheme == "risk_parity":
        w = risk_parity(returns)
    elif scheme == "hrp":
        w = hrp(returns)
    elif scheme == "mvo":
        w = mvo(returns, scores, max_weight=max_weight, **kwargs)
    else:
        w = black_litterman(
            returns, scores, market_caps=market_caps, max_weight=max_weight, **kwargs
        )
    return cap_and_normalize(w, max_weight)


def cap_sector_weights(
    weights: pd.Series,
    sectors: pd.Series,
    max_sector_weight: float,
) -> pd.Series:
    """
    섹터 비중 상한 — water-filling 을 섹터 단위로 적용한다.

    `neutralize=("sector",)` 는 **스코어**에서 섹터 효과를 뺀다. 그러나 상위 N을
    고르고 나면 결과가 한 섹터에 몰릴 수 있고, 그건 팩터 베팅이 아니라
    **의도하지 않은 매크로 베팅**이다 — 금리가 움직이면 그 하나로 성과가
    결정된다. 실측: 채택 전략 보유 19종목 중 6종목(32%)이 Technology 였다.

    섹터를 자르되 **그 안의 상대 비중은 유지한다** — 스코어 순서를 뒤집지
    않기 위해서다. 섹터 수 × 상한 < 1 이라 만족이 불가능하면 원본을 그대로
    돌려준다 (조용히 이상한 값을 내지 않는다). 섹터 미상 종목은 각각을
    독립 버킷으로 본다 — 한 덩어리로 묶어 잘라내면 근거 없는 제약이 된다.
    """
    if max_sector_weight >= 1.0 or weights.empty:
        return weights

    labels = sectors.reindex(weights.index)
    unknown = labels.isna()
    # 섹터 미상은 각자 고유 라벨을 주어 서로 묶이지 않게 한다
    labels = labels.astype(object).where(~unknown, pd.Series(weights.index, index=weights.index))

    if labels.nunique() * max_sector_weight < 1.0 - 1e-9:
        return weights  # 만족 불가능한 상한 — 원본 유지

    w = weights.clip(lower=0.0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(w), index=w.index)

    for _ in range(64):
        totals = w.groupby(labels).transform("sum")
        over = totals > max_sector_weight + 1e-12
        if not over.any():
            break
        # 초과 섹터는 상한까지 비례 축소, 그 몫을 나머지에 비례 배분
        w = w.where(~over, w * max_sector_weight / totals)
        slack = 1.0 - w.sum()
        free = ~over & (w > 0)
        if slack <= 1e-12 or not free.any():
            break
        w = w.where(~free, w + slack * w / w[free].sum())
    return w / w.sum()
