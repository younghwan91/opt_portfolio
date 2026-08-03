"""
가치 팩터 — EV 관련 (9개)

EV = 시가총액 + 총차입금 − 현금성자산

퀀트 관점:
- EV 배수는 자본구조 중립이라 차입 수준이 제각각인 유니버스에서
  PER/PBR 보다 비교 가능성이 높다.
- 순현금 기업(현금 > 차입금 + 시총)은 EV 가 음수가 되어 배수 부호가
  뒤집힌다. `EV <= 0` 은 NaN 으로 떨어뜨린다.
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.dsl.registry import factor

#: 벤더 제공 `ev` 대신 자체 계산 — 벤더 간 정의 차이를 흡수한다.
EV_EXPR = F.mcap + F.debt - F.cashneq

EV = factor("EV", EV_EXPR, category="value_ev", label="EV", direction=-1,
            requires=("SF1", "SEP"))

#: (팩터명, 분모 표현식, 한글 라벨, 비고)
_EV_MULTIPLES = [
    ("EV_NET", F.netinc, "EV/Net", ""),
    ("EV_SALES", F.revenue, "EV/Sales", ""),
    ("EV_EBITDA", F.ebitda, "EV/EBITDA", "가장 널리 쓰이는 EV 배수"),
    ("EV_EBIT", F.ebit, "EV/EBIT", ""),
    ("EV_GP", F.gp, "EV/GP", "매출총이익 기준 — 회계 재량이 가장 적음"),
    ("EV_RD", F.rnd, "EV/R&D", "R&D 미지출 기업 다수 → NaN"),
    ("EV_CF", F.ncfo, "EV/CF", ""),
    ("EV_AC", F.netinc - F.ncfo, "EV/AC", "발생액 부호 불안정 — 해석 주의"),
]

EV_FACTORS = {
    name: factor(
        name,
        EV_EXPR / denom,
        category="value_ev",
        label=label,
        invert=True,
        neutralize=("sector",) if name == "EV_RD" else (),
        requires=("SF1", "SEP"),
        notes=notes,
    )
    for name, denom, label, notes in _EV_MULTIPLES
}

__all__ = ["EV", "EV_EXPR", "EV_FACTORS"]
