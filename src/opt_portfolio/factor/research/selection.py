"""
학습 구간 내 팩터 선택 — 조합 탐색을 정직하게 만드는 장치.

팩터 158개 중 5개를 고르는 조합은 8억 가지다. 사후에 최고를 고르면
백테스트는 반드시 좋아지고, DSR 은 그 시도 횟수를 벌해 결과를 0 으로
만든다. 실제로 이 저장소에서 시도 228회(DSR 0.402)를 57회(0.910)로
줄이자 성과가 올랐다 — **시도를 줄이는 것이 성과를 올리는 길이었다.**

해법은 선택을 학습 구간 안으로 옮기는 것이다. walk-forward 의 각 폴드에서
학습 데이터로만 팩터를 고르고 그 조합으로 검증 구간을 한 번 실행하면,
조합 탐색 자체가 OOS 로 검증된다. 폴드마다 선택이 달라지는 것 자체도
정보다 — 안정적으로 같은 팩터가 뽑히면 그것이 진짜 신호라는 증거다.
"""

from __future__ import annotations

import pandas as pd

#: IC 가 확정되기까지의 지연(개월). 순방향 수익 21일 ≈ 1개월 + 여유 1개월.
DEFAULT_LAG = 2

#: 횡단면 회귀에 필요한 최소 종목 수. 이보다 적으면 잔차가 표본 잡음이다.
MIN_CROSS_SECTION_FOR_REGRESSION = 30


def select_factors(
    panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    end: pd.Timestamp,
    *,
    k: int = 5,
    min_ic: float = 0.0,
    min_months: int = 36,
    lag: int = DEFAULT_LAG,
) -> list[str]:
    """
    `end` 시점까지 관측 가능한 정보만으로 상위 k개 팩터를 고른다.

    Args:
        panels: {팩터명: (date × ticker) 스코어 패널}
        forward_returns: 같은 그리드의 순방향 수익
        end: 학습 구간의 끝 — 이 이후 데이터는 보지 않는다
        k: 고를 팩터 수
        min_ic: 이 값 이하의 평균 IC 는 담지 않는다 (기본 0 = 양수만)
        min_months: 이보다 관측이 적으면 선택하지 않고 전체를 쓴다
        lag: IC 확정 지연(개월). 순방향 수익을 쓰므로 최근 구간의 IC 는
            아직 알 수 없다 — 그만큼 잘라낸다.

    Returns:
        팩터명 리스트. 근거가 부족하면 **전체 목록**(= 1/N 후퇴).
    """
    from opt_portfolio.factor.research.ic import rank_ic

    if not panels:
        return []

    usable = forward_returns.index[forward_returns.index <= end]
    if lag > 0:
        usable = usable[:-lag] if len(usable) > lag else usable[:0]
    if len(usable) < min_months:
        return list(panels)  # 근거 부족 → 전체 사용

    fwd = forward_returns.loc[usable]
    scores: dict[str, float] = {}
    for name, panel in panels.items():
        ic = rank_ic(panel.reindex(usable), fwd)
        mean_ic = float(ic.mean())
        if pd.notna(mean_ic):
            scores[name] = mean_ic

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    picked = [name for name, ic in ranked[:k] if ic > min_ic]
    return picked or list(panels)  # 아무것도 통과 못하면 전체 사용


def _residualize(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """
    같은 날짜의 횡단면에서 `candidates` 를 `selected` 에 회귀하고 잔차를 돌려준다.

    두 프레임 모두 (ticker × 팩터) 이며, **양쪽이 함께 관측된 종목**만 쓴다.
    한쪽만 있는 종목을 0 으로 채우면 없는 정보를 만들어내는 것이다.
    """
    import numpy as np

    usable = selected.notna().all(axis=1) & candidates.notna().any(axis=1)
    out = pd.DataFrame(np.nan, index=candidates.index, columns=candidates.columns)
    if usable.sum() < MIN_CROSS_SECTION_FOR_REGRESSION:
        return out

    x = selected.loc[usable].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])  # 절편 — 평균 차이는 중복이 아니다
    for col in candidates.columns:
        y = candidates.loc[usable, col]
        ok = y.notna().to_numpy()
        if ok.sum() < MIN_CROSS_SECTION_FOR_REGRESSION:
            continue
        beta, *_ = np.linalg.lstsq(x[ok], y.to_numpy(dtype=float)[ok], rcond=None)
        resid = y.to_numpy(dtype=float)[ok] - x[ok] @ beta
        out.loc[y.index[ok], col] = resid
    return out


def select_factors_residual(
    panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    end: pd.Timestamp,
    *,
    k: int = 5,
    min_ic: float = 0.0,
    min_months: int = 36,
    lag: int = DEFAULT_LAG,
) -> list[str]:
    """
    **잔차 기여도**로 고르는 전진 선택 — 중복을 빼고 남는 정보만 센다.

    왜 필요한가 (실측): 개별 IC 상위 6개를 고르는 방식(`select_factors`)은
    19개 폴드 전부에서 가치·품질 팩터만 뽑았고, 성장 팩터 4종은 **한 번도**
    선택되지 않았다. 그런데 성장을 포함한 고정 조합이 더 좋았다
    (CAGR 16.90% vs 15.83%).

    개별 IC 는 팩터를 하나씩 세워놓고 재는 자다. 가치·품질 팩터는 서로 비슷한
    종목을 고르므로 IC 가 높아도 **새로 더하는 정보는 적다.** IC 순으로 자르면
    겹치는 것들이 자리를 다 차지하고 다른 축이 통째로 사라진다.

    그래서 한 개씩 고르되, 매번 **이미 고른 것들에 회귀하고 남은 잔차**의 IC 로
    다음을 정한다. 첫 팩터는 개별 IC 최대, 그 다음부터는 기여도 최대다.

    Returns:
        팩터명 리스트. 근거가 부족하면 **전체 목록**(= 1/N 후퇴).
    """
    from opt_portfolio.factor.portfolio.score import rank_normalize
    from opt_portfolio.factor.research.ic import rank_ic

    if not panels:
        return []

    usable = forward_returns.index[forward_returns.index <= end]
    if lag > 0:
        usable = usable[:-lag] if len(usable) > lag else usable[:0]
    if len(usable) < min_months:
        return list(panels)

    fwd = forward_returns.loc[usable]
    # 잔차를 재려면 스케일이 같아야 한다 — 회귀계수가 단위 차이를 흡수해버리면
    # "겹친다"와 "크다"를 구분할 수 없다.
    normalized = {name: rank_normalize(p.reindex(usable)) for name, p in panels.items()}

    chosen: list[str] = []
    remaining = dict(normalized)
    while remaining and len(chosen) < k:
        if not chosen:
            contribution = {n: float(rank_ic(p, fwd).mean()) for n, p in remaining.items()}
        else:
            resid: dict[str, list[pd.Series]] = {n: [] for n in remaining}
            for date in usable:
                sel = pd.concat([normalized[n].loc[date] for n in chosen], axis=1, keys=chosen)
                cand = pd.concat(
                    [remaining[n].loc[date] for n in remaining], axis=1, keys=list(remaining)
                )
                r = _residualize(cand, sel)
                for n in remaining:
                    resid[n].append(r[n].rename(date))
            contribution = {
                n: float(rank_ic(pd.DataFrame(rows).reindex(usable), fwd).mean())
                for n, rows in resid.items()
            }

        best = max(contribution, key=lambda n: contribution[n])
        if not pd.notna(contribution[best]) or contribution[best] <= min_ic:
            break
        chosen.append(best)
        remaining.pop(best)

    return chosen or list(panels)


#: 설정 파일의 `select_method` 값 → 선별 함수. 오타는 KeyError 로 즉시 죽는다
#: — 조용히 기본값으로 후퇴하면 어느 기준으로 돈 실험인지 알 수 없게 된다.
SELECTORS = {
    "ic": select_factors,
    "residual": select_factors_residual,
}
