"""
성장 팩터 (26개) — 전부 자동 생성

이 파일에는 성장률을 계산하는 코드가 한 줄도 없다.
base 표현식 13개를 나열하면 derive_growth() 가 QoQ/YoY 쌍을 만든다.
13 × 2 = 26개.

퀀트 관점:
- 성장률 분모에 abs() 를 쓰는 처리는 DSL 의 Growth 노드에 있다.
  적자 축소(-100 → -50)를 성장으로 잡기 위함이며, 이 처리가 없으면
  턴어라운드 기업의 성장률 부호가 통째로 뒤집힌다.
- 성장 팩터는 섹터 편향이 크다(바이오는 항상 고성장). 기본 중립화를 건다.
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.dsl.registry import derive_growth

#: (base 표현식, 팩터명 stem, 한글 라벨, 필요 데이터셋)
_GROWTH_BASES = [
    (F.netinc, "NETINC", "순이익", ("SF1",)),
    (F.opinc, "OPINC", "영업이익", ("SF1",)),
    (F.gp, "GP", "매출총이익", ("SF1",)),
    (F.revenue, "REVENUE", "매출액", ("SF1",)),
    (F.assets, "ASSETS", "자산", ("SF1",)),
    (F.equity, "EQUITY", "자본", ("SF1",)),
    (F.gp / F.assets, "GP_A", "GP/A", ("SF1",)),
    (F.opinc / F.debt, "OPINC_DEBT", "영업이익 / 차입금 ", ("SF1",)),
    (F.ncfo, "NCFO", "현금흐름 ", ("SF1",)),
    (F.rnd, "RND", "연구개발비 지출 ", ("SF1",)),
    (F.cashneq, "CASH", "보유 현금성자산 ", ("SF1",)),
    (F.debt, "DEBT", "차입금 ", ("SF1",)),
    ((-F.ncfdiv) / F.mcap, "DIVYIELD", "배당수익률 ", ("SF1", "SEP")),
]

#: {stem: {"qoq": FactorSpec, "yoy": FactorSpec}}
GROWTH_FACTORS = {
    stem: derive_growth(expr, stem, label, requires=requires, neutralize=("sector",))
    for expr, stem, label, requires in _GROWTH_BASES
}

__all__ = ["GROWTH_FACTORS"]
