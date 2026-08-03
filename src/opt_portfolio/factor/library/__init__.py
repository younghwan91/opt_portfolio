"""
팩터 라이브러리 — import 시 전역 레지스트리에 모든 팩터가 등록된다.

모듈을 import 하는 것만으로 등록이 일어나므로, 순서와 무관하게
`from opt_portfolio.factor.library import REGISTRY` 후 전체 조회가 가능하다.
"""

from opt_portfolio.factor.dsl.registry import REGISTRY

from . import (  # noqa: F401  (import 부수효과로 팩터가 등록됨)
    acceleration,
    flow_proxy,
    growth,
    price,
    quality,
    value_ev,
    value_price,
)

__all__ = [
    "REGISTRY",
    "acceleration",
    "flow_proxy",
    "growth",
    "price",
    "quality",
    "value_ev",
    "value_price",
]
