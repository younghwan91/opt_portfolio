"""
팩터 레지스트리

팩터를 이름·카테고리·메타와 함께 등록하고, TTM/성장/가속 파생형을
일괄 생성하는 헬퍼를 제공한다.

퀀트 관점:
- `direction` 과 `invert` 를 메타로 분리한 이유: PER 을 그대로 오름차순
  정렬하면 적자기업(음수 PER)이 최상위로 온다. 배수형 팩터는 역수
  (earnings yield 형태)로 스코어링하고 표시만 배수로 한다.
- `requires` 로 미구독 데이터셋 의존 팩터를 자동 비활성화한다.
  에러로 죽는 대신 경고 후 skip 하여 부분 구독 상태에서도 돌아가게 한다.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Literal

from opt_portfolio.factor.dsl.context import PanelContext
from opt_portfolio.factor.dsl.expr import Expr, Panel

logger = logging.getLogger(__name__)

Category = Literal[
    "value_price",
    "value_ev",
    "quality",
    "price",
    "growth",
    "acceleration",
    "flow_proxy",
]


@dataclass(frozen=True)
class FactorSpec:
    """팩터 하나의 완전한 정의."""

    name: str
    category: Category
    expr: Expr
    label: str = ""
    direction: int = 1
    invert: bool = False
    neutralize: tuple[str, ...] = ()
    winsor: float = 0.01
    requires: frozenset[str] = dc_field(default_factory=lambda: frozenset({"SF1"}))
    derived_from: str | None = None
    notes: str = ""

    def scoring_expr(self) -> Expr:
        """
        랭킹에 실제로 투입되는 표현식.

        배수형(invert=True)은 역수를 취해 '클수록 좋음'으로 방향을 통일하고,
        중립화·윈저라이즈를 선언된 순서대로 적용한다.
        """
        expr = self.expr
        if self.invert:
            expr = 1.0 / expr
        if self.winsor:
            expr = expr.winsor(self.winsor)
        if self.neutralize:
            expr = expr.neutralize(self.neutralize)
        if self.direction < 0:
            expr = -expr
        return expr

    def evaluate(self, ctx: PanelContext, *, scoring: bool = False) -> Panel:
        return (self.scoring_expr() if scoring else self.expr).eval(ctx)

    def formula(self) -> str:
        return self.expr.describe()


class FactorRegistry:
    """이름 → FactorSpec 매핑. 중복 등록을 에러로 잡는다."""

    def __init__(self) -> None:
        self._specs: dict[str, FactorSpec] = {}

    def register(self, spec: FactorSpec) -> FactorSpec:
        if spec.name in self._specs:
            raise DuplicateFactorError(f"팩터 '{spec.name}' 이(가) 이미 등록되어 있습니다")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> FactorSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownFactorError(f"등록되지 않은 팩터: '{name}'") from None

    def all(self) -> list[FactorSpec]:
        return list(self._specs.values())

    def by_category(self, category: Category) -> list[FactorSpec]:
        return [s for s in self._specs.values() if s.category == category]

    def available(self, subscribed: Iterable[str]) -> list[FactorSpec]:
        """구독 중인 데이터셋만으로 계산 가능한 팩터."""
        have = frozenset(subscribed)
        usable, skipped = [], []
        for spec in self._specs.values():
            (usable if spec.requires <= have else skipped).append(spec)
        if skipped:
            logger.warning(
                "미구독 데이터셋 의존으로 %d개 팩터를 비활성화합니다: %s",
                len(skipped),
                ", ".join(sorted(s.name for s in skipped)[:10]),
            )
        return usable

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs


REGISTRY = FactorRegistry()


def factor(
    name: str,
    expr: Expr,
    *,
    category: Category,
    label: str = "",
    direction: int = 1,
    invert: bool = False,
    neutralize: tuple[str, ...] = (),
    winsor: float = 0.01,
    requires: Iterable[str] = ("SF1",),
    derived_from: str | None = None,
    notes: str = "",
) -> FactorSpec:
    """팩터를 전역 레지스트리에 등록한다."""
    return REGISTRY.register(
        FactorSpec(
            name=name,
            category=category,
            expr=expr,
            label=label or name,
            direction=direction,
            invert=invert,
            neutralize=tuple(neutralize),
            winsor=winsor,
            requires=frozenset(requires),
            derived_from=derived_from,
            notes=notes,
        )
    )


# ------------------------------------------------------------------ 자동 파생


def derive_ttm(base: FactorSpec, *, name: str | None = None, label: str = "") -> FactorSpec:
    """base 팩터의 TTM 변형을 등록한다."""
    return REGISTRY.register(
        FactorSpec(
            name=name or f"{base.name}_TTM",
            category=base.category,
            expr=base.expr.ttm(),
            label=label or f"{base.label} (TTM)",
            direction=base.direction,
            invert=base.invert,
            neutralize=base.neutralize,
            winsor=base.winsor,
            requires=base.requires,
            derived_from=base.name,
        )
    )


def derive_growth(
    base_expr: Expr,
    stem: str,
    label_ko: str,
    *,
    category: Category = "growth",
    requires: Iterable[str] = ("SF1",),
    neutralize: tuple[str, ...] = (),
) -> dict[str, FactorSpec]:
    """
    QoQ / YoY 성장 팩터 한 쌍을 자동 등록한다.

    성장 팩터 26개가 전부 이 함수 호출 13번으로 만들어진다.
    """
    return {
        "qoq": factor(
            f"{stem}_GROWTH_QOQ",
            base_expr.qoq(),
            category=category,
            label=f"{label_ko}성장률 (QoQ)",
            requires=requires,
            neutralize=neutralize,
        ),
        "yoy": factor(
            f"{stem}_GROWTH_YOY",
            base_expr.yoy(),
            category=category,
            label=f"{label_ko}성장률 (YoY)",
            requires=requires,
            neutralize=neutralize,
        ),
    }


def derive_acceleration(
    base_expr: Expr,
    stem: str,
    label_ko: str,
    *,
    requires: Iterable[str] = ("SF1",),
) -> dict[str, FactorSpec]:
    """
    성장 가속 팩터 한 쌍(QoQ/YoY)을 자동 등록한다.

    가속 = 성장률의 차분(2차 미분). 노이즈가 2번 증폭되므로 윈저라이즈를
    기본 팩터보다 강하게(2%) 건다.
    """
    return {
        "qoq": factor(
            f"{stem}_ACCEL_QOQ",
            base_expr.qoq().accel(periods=1),
            category="acceleration",
            label=f"{label_ko}성장 가속 (QoQ)",
            winsor=0.02,
            requires=requires,
            notes="2차 차분 — 노이즈 증폭에 주의",
        ),
        "yoy": factor(
            f"{stem}_ACCEL_YOY",
            base_expr.yoy().accel(periods=4),
            category="acceleration",
            label=f"{label_ko}성장 가속 (YoY)",
            winsor=0.02,
            requires=requires,
            notes="2차 차분 — 노이즈 증폭에 주의",
        ),
    }


class DuplicateFactorError(ValueError):
    """같은 이름의 팩터를 두 번 등록했을 때."""


class UnknownFactorError(KeyError):
    """등록되지 않은 팩터를 조회했을 때."""
