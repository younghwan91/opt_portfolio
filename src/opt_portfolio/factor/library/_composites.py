"""
복합 스코어 — Piotroski F-score, Altman Z-score

이진 조건을 표현식 트리로 다루기 위한 Predicate 노드를 여기서 정의한다.
산술 연산만으로는 "ROA > 0" 같은 조건을 표현할 수 없기 때문.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from opt_portfolio.factor.dsl.context import PanelContext
from opt_portfolio.factor.dsl.expr import Expr, F, Panel


@dataclass(frozen=True)
class Predicate(Expr):
    """조건 충족 시 1.0, 미충족 시 0.0, 원본이 NaN 이면 NaN 을 유지한다."""

    child: Expr
    test: Callable[[pd.DataFrame], pd.DataFrame]
    label: str

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        flags = self.test(panel.data).astype("float64")
        return panel.with_data(flags.where(panel.data.notna()))

    def describe(self) -> str:
        return f"{self.label}({self.child.describe()})"


def is_positive(expr: Expr) -> Expr:
    return Predicate(expr, lambda df: df > 0, "positive")


def increased_yoy(expr: Expr) -> Expr:
    """전년동기 대비 증가했는가. 4분기 차분의 부호."""
    return Predicate(expr - expr.lag(4), lambda df: df > 0, "increased")


def decreased_yoy(expr: Expr) -> Expr:
    return Predicate(expr - expr.lag(4), lambda df: df < 0, "decreased")


def piotroski_f_expr() -> Expr:
    """
    Piotroski F-score (0~9).

    수익성 4 + 레버리지/유동성 3 + 운영효율 2.
    각 항목은 독립적으로 NaN 이 될 수 있으므로, 합산 시 한 항목이라도
    NaN 이면 전체가 NaN 이 된다. 이는 의도된 동작이다 —
    일부만 계산된 F-score 는 다른 종목의 F-score 와 비교 불가능하다.
    """
    roa = F.netinc / F.assetsavg
    leverage = F.debtnc / F.assetsavg
    current = F.assetsc / F.liabilitiesc
    gross_margin = F.gp / F.revenue
    asset_turnover = F.revenue / F.assetsavg

    return (
        # 수익성
        is_positive(roa)
        + is_positive(F.ncfo)
        + increased_yoy(roa)
        + is_positive(F.ncfo - F.netinc)  # 발생액 품질: CFO > 순이익
        # 레버리지 · 유동성 · 자금조달
        + decreased_yoy(leverage)
        + increased_yoy(current)
        + Predicate(F.sharesbas - F.sharesbas.lag(4), lambda df: df <= 0, "no_issuance")
        # 운영 효율
        + increased_yoy(gross_margin)
        + increased_yoy(asset_turnover)
    )


def altman_z_expr() -> Expr:
    """
    Altman Z-score (제조업 원식).

        Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5

    | 항목 | 정의 |
    |---|---|
    | X1 | 운전자본 / 총자산 |
    | X2 | 이익잉여금 / 총자산 |
    | X3 | EBIT / 총자산 |
    | X4 | 시가총액 / 총부채 |
    | X5 | 매출액 / 총자산 |

    금융업은 자본구조가 근본적으로 달라 이 공식이 무의미하다.
    금융주 제외 유니버스를 전제로 한다.
    """
    x1 = F.workingcapital / F.assets
    x2 = F.retearn / F.assets
    x3 = F.ebit / F.assets
    x4 = F.mcap / F.liabilities
    x5 = F.revenue / F.assets
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


#: Altman Z 판정 구간 — 유니버스 필터('관리종목 제외' 대체)에서 참조
ALTMAN_DISTRESS = 1.81
ALTMAN_SAFE = 2.99
