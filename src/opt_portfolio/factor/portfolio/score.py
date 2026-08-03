"""
멀티팩터 스코어 합성

퀀트 관점:
- 팩터별 원점수는 스케일이 제각각이라 (PER 역수 vs 모멘텀 %) 합성 전
  반드시 횡단면 정규화가 필요하다. 기본은 rank → 정규분포 역변환
  (rank-normal): z-score 보다 극단값에 강건하고, 팩터 간 기여가
  분포 모양이 아니라 순위로만 결정된다.
- 두 단계 합성 (팩터 → 카테고리 → 최종): 가치 팩터 31개를 낱개로
  가중하면 가치가 31표, 사이즈가 1표를 갖는 왜곡이 생긴다.
  카테고리 내부에서 먼저 평균해 카테고리당 1표로 만든다.
- IC 가중은 look-ahead 위험이 있다 — 반드시 **트레일링** IC 만 쓰고,
  음수 IC 는 0 으로 클립한다 (부호 반전 베팅은 별도 결정이어야 한다).
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from opt_portfolio.factor.dsl.registry import FactorSpec


def rank_normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """횡단면 rank → 표준정규 역변환. 극단값에 강건한 정규화."""
    pct = panel.rank(axis=1, pct=True)
    n = panel.notna().sum(axis=1)
    # (r − 0.5)/n 꼴로 0/1 경계를 피한다
    adj = pct.sub(0.5 / n, axis=0).clip(1e-6, 1 - 1e-6)
    return pd.DataFrame(
        stats.norm.ppf(adj), index=panel.index, columns=panel.columns
    ).where(panel.notna())


def composite_score(
    panels: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    *,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    팩터 패널들을 정규화 후 가중 평균한다.

    Args:
        panels: {팩터명: (date × ticker) 스코어 패널}. FactorSpec.evaluate(
            scoring=True) 결과를 넣으면 방향·역수 처리가 이미 끝나 있다.
        weights: {팩터명: 가중치}. 미지정 시 균등.
        min_coverage: 종목별로 관측된 팩터 가중치 합이 이 비율 미만이면
            그 (날짜, 종목) 은 NaN — 팩터 1개만 있는 종목이 팩터 15개인
            종목과 같은 신뢰도로 랭킹되는 것을 막는다.
    """
    if not panels:
        raise ValueError("합성할 팩터 패널이 없습니다")
    w = weights or {name: 1.0 for name in panels}
    total_w = sum(abs(v) for v in w.values())

    num, cov = None, None
    for name, panel in panels.items():
        z = rank_normalize(panel) * w.get(name, 0.0)
        mask = panel.notna() * abs(w.get(name, 0.0))
        num = z.fillna(0.0) if num is None else num.add(z.fillna(0.0), fill_value=0.0)
        cov = mask if cov is None else cov.add(mask, fill_value=0.0)

    score = num / cov.where(cov > 0)
    return score.where(cov >= min_coverage * total_w)


def composite_by_category(
    panels: dict[str, pd.DataFrame],
    specs: dict[str, FactorSpec],
    category_weights: dict[str, float],
) -> pd.DataFrame:
    """
    두 단계 합성: 카테고리 내부 균등 평균 → 카테고리 간 가중 평균.

    category_weights 의 키가 곧 사용할 카테고리다 — 여기에 없는
    카테고리의 팩터는 무시된다. 이 가중치 벡터가 PO 의 최상위 축이다.
    """
    by_cat: dict[str, dict[str, pd.DataFrame]] = {}
    for name, panel in panels.items():
        cat = specs[name].category
        if cat in category_weights:
            by_cat.setdefault(cat, {})[name] = panel

    cat_scores = {
        cat: composite_score(members) for cat, members in by_cat.items()
    }
    return composite_score(cat_scores, dict(category_weights))


def trailing_ic_weights(
    ic_series: dict[str, pd.Series],
    as_of: pd.Timestamp,
    window: int = 756,
) -> dict[str, float]:
    """
    트레일링 평균 IC 비례 가중 (음수는 0 클립).

    as_of 이전 데이터만 사용한다 — 이 함수를 전체 표본 IC 로 호출하는 순간
    합성 자체가 look-ahead 가 된다. 백테스트 안에서는 리밸런싱 날짜마다
    호출해야 한다.
    """
    weights = {}
    for name, ic in ic_series.items():
        past = ic.loc[:as_of].tail(window)
        weights[name] = float(max(past.mean(), 0.0)) if len(past) >= 60 else 0.0

    total = sum(weights.values())
    if total <= 0:
        return {name: 1.0 / len(weights) for name in weights}
    return {name: v / total for name, v in weights.items()}
