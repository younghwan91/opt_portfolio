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
