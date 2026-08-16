"""전술적 자산배분(TAA) — ETF 월별 로테이션.

`factor/`(개별주 팩터) 와도 `strategies/`(기존 VAA) 와도 분리된 새 서브시스템이다.
설계 근거는 `docs/superpowers/specs/2026-08-17-taa-strategy-design.md`.
"""

from .data import load_prices

__all__ = ["load_prices"]
