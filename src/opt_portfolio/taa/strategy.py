"""전략을 데이터로 선언한다.

VAA·BAA·정적 배분이 **같은 엔진을 타야** 비교가 성립한다. 그래서 전략을
코드가 아니라 `StrategySpec` 으로 표현한다 — 팩터 엔진이 전략을 JSON 하나로
선언하는 것과 같은 이유다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """전략 하나의 완전한 선언.

    Attributes:
        canary: 위험 경보용 자산. **투자 대상과 분리한다** — VAA 는 이 둘을
            겸해서, 살 생각도 없는 EEM·EFA 약세가 포트폴리오를 방어로 밀어냈다.
        offensive: 위험 국면이 아닐 때 고를 후보
        defensive: 위험 국면에 고를 후보
        selection: 선택 지표. `"13612w"`(VAA) 또는 `"sma13"`(BAA)
        cash_ticker: 지정하면 방어 자산이 이것보다 못할 때 현금으로 대체한다
            (dual momentum). `None` 이면 비활성.
        static_weights: 지정하면 시그널을 무시하고 이 비중을 유지한다 (기준선용)
    """

    name: str
    canary: tuple[str, ...]
    offensive: tuple[str, ...]
    defensive: tuple[str, ...]
    top_n_offensive: int
    top_n_defensive: int
    selection: str = "sma13"
    cash_ticker: str | None = None
    static_weights: dict[str, float] | None = None

    def tickers(self) -> list[str]:
        """이 전략이 필요로 하는 전체 티커."""
        names = set(self.canary) | set(self.offensive) | set(self.defensive)
        if self.cash_ticker:
            names.add(self.cash_ticker)
        if self.static_weights:
            names |= set(self.static_weights)
        return sorted(names)


def is_defensive(spec: StrategySpec, mom: pd.DataFrame, date: pd.Timestamp) -> bool:
    """카나리아 중 **하나라도** 모멘텀이 음수면 방어 (breadth 규칙)."""
    if not spec.canary:
        return False
    scores = mom.loc[date, list(spec.canary)]
    return bool((scores < 0).any())


def select_weights(
    spec: StrategySpec,
    mom: pd.DataFrame,
    sel: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """해당 시점의 목표 비중. 합은 1.0 (전액 현금 대체 시에도)."""
    if spec.static_weights is not None:
        return dict(spec.static_weights)

    metric = mom if spec.selection == "13612w" else sel
    defensive = is_defensive(spec, mom, date)
    pool = list(spec.defensive) if defensive else list(spec.offensive)
    top_n = spec.top_n_defensive if defensive else spec.top_n_offensive

    ranked = metric.loc[date, pool].sort_values(ascending=False)
    picks = list(ranked.index[:top_n])

    # dual momentum — 현금을 못 이기는 방어 자산은 현금으로 바꾼다.
    # SHY 를 연 0.05% 로 24% 기간 들고 있던 문제의 해법이다.
    if defensive and spec.cash_ticker:
        cash_score = metric.loc[date, spec.cash_ticker]
        picks = [t if metric.loc[date, t] > cash_score else spec.cash_ticker for t in picks]

    weight = 1.0 / len(picks)
    out: dict[str, float] = {}
    for t in picks:
        out[t] = out.get(t, 0.0) + weight
    return out
