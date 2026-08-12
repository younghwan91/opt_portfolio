"""
실험 팩터 — 라이브러리에 없는 조합을 문헌 근거로 직접 조립한다.

여기 있는 팩터는 **아직 검증되지 않았다**. 팩터 연구소(`scripts/factor_lab.py`)가
10분할 테스트와 IC 로 걸러내며, 통과한 것만 정식 카테고리로 승격한다.

원칙:
- 사후에 좋아 보여서 넣지 않는다. 각 팩터에는 **왜 작동해야 하는지**를 먼저 적는다.
  이유를 못 대는 팩터는 데이터 마이닝이고, DSR 이 그 비용을 청구한다.
- 회계 항목의 **차분**은 `x - x.lag(1)` 로 쓴다. `.qoq()` 는 증감'률'이라
  분모가 0 이나 음수인 항목(운전자본·차입금)에서 폭발한다.
- 섹터 편향이 심한 항목(R&D·무형자산·설비투자)은 `neutralize=("sector",)` 를 건다.
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import Expr, F
from opt_portfolio.factor.dsl.registry import factor

_SEP = ("SF1", "SEP")


def _delta(expr: Expr, periods: int = 4) -> Expr:
    """전년동기 대비 차분 — 계절성을 피하려 4분기 전과 비교한다."""
    return expr - expr.lag(periods)


# ------------------------------------------------------------------ 수익성
#: 현금 기반 영업수익성 (Ball·Gerakos·Linnainmaa·Nikolaev 2016).
#: 총이익에서 발생액(매출채권·재고 증가)을 빼면 '아직 현금이 아닌 이익'이
#: 제거된다. 논문은 이 팩터가 GP_A 를 지배한다고 보고한다 — 우리 IC 스캔에서
#: GP_A 가 가장 깨끗했으므로, 그 개선판이 실제로 더 나은지가 핵심 질문이다.
CBOP = factor(
    "CBOP",
    (F.gp - _delta(F.receivables) - _delta(F.inventory) + _delta(F.liabilitiesc)) / F.assets,
    category="quality",
    label="현금기반 영업수익성",
    notes="Ball et al. 2016 — GP_A 의 발생액 제거판",
)

#: 발생액 (Sloan 1996). 이익이 현금흐름보다 크면 그 차이는 회계적 발생액이고,
#: 시장은 그 지속성을 과대평가한다. 낮을수록 좋다.
ACCRUAL_CF = factor(
    "ACCRUAL_CF",
    (F.netinc - F.ncfo) / F.assets,
    category="quality",
    label="발생액 (순이익−영업현금)",
    direction=-1,
    notes="Sloan 1996 — 발생액이 큰 기업의 이익은 덜 지속된다",
)

#: 순영업자산 (Hirshleifer·Hou·Teoh·Zhang 2004). 누적 발생액의 저량(stock)
#: 버전이다. 영업자산이 부풀어 있으면 과거 이익이 현금이 아니었다는 뜻.
NOA = factor(
    "NOA",
    ((F.assets - F.cashneq) - (F.liabilities - F.debt)) / F.assets.lag(4),
    category="quality",
    label="순영업자산 비율",
    direction=-1,
    notes="Hirshleifer et al. 2004 — 발생액의 누적 저량",
)

# ------------------------------------------------------------------ 자금조달
#: 순주식발행 (Daniel·Titman 2006). 주식을 찍어내는 기업은 이후 수익률이 낮다.
#: 경영진이 고평가 시점에 발행한다는 해석과, 발행 자체가 희석이라는 해석 둘 다 있다.
NET_ISSUANCE = factor(
    "NET_ISSUANCE",
    _delta(F.sharesbas) / F.sharesbas.lag(4),
    category="quality",
    label="순주식발행",
    direction=-1,
    notes="Daniel & Titman 2006 — 발행 기업의 이후 수익률이 낮다",
)

#: 외부 자금조달 총액 (Bradshaw·Richardson·Sloan 2006). 주식·차입을 합친
#: 순조달이 클수록 이후 수익률이 낮다. 위 발행 팩터의 차입 포함판이다.
EXT_FINANCE = factor(
    "EXT_FINANCE",
    (F.ncfcommon + F.ncfdebt) / F.assets,
    category="quality",
    label="외부 자금조달",
    direction=-1,
    notes="Bradshaw et al. 2006",
)

# ------------------------------------------------------------------ 투자
#: 설비투자 강도 (Titman·Wei·Xie 2004). 과잉투자 기업의 이후 수익률이 낮다.
#: Sharadar 의 capex 는 음수이므로 부호를 뒤집어 '투자 강도'로 만든다.
CAPEX_INTENSITY = factor(
    "CAPEX_INTENSITY",
    -F.capex / F.assets,
    category="quality",
    label="설비투자 강도",
    direction=-1,
    neutralize=("sector",),
    notes="Titman et al. 2004 — 섹터별 자본집약도 차이가 커 중립화 필수",
)

# ------------------------------------------------------------------ 무형자산
#: R&D 집약도 (Chan·Lakonishok·Sougiannis 2001). 회계는 R&D 를 비용 처리하므로
#: 장부가가 과소평가되고, 시장은 그 가치를 늦게 반영한다. 섹터 편향이 극심하다.
RND_MCAP = factor(
    "RND_MCAP",
    F.rnd.ttm() / F.mcap,
    category="value_price",
    label="R&D / 시가총액",
    neutralize=("sector",),
    requires=_SEP,
    notes="Chan et al. 2001 — 비용 처리되는 무형투자의 가치",
)

#: 주식보상 부담. R&D 집약 기업이 SBC 로 비용을 이연하는 정도.
#: 문헌보다는 현대 미장의 실무 관찰에 기댄 팩터라 검증이 특히 중요하다.
SBC_REVENUE = factor(
    "SBC_REVENUE",
    F.sbcomp.ttm() / F.revenue.ttm(),
    category="quality",
    label="주식보상 / 매출",
    direction=-1,
    neutralize=("sector",),
    notes="검증 필요 — 문헌 근거가 위 팩터들보다 약하다",
)

# ------------------------------------------------------------------ 운영 효율
#: 판관비 효율. 같은 총이익을 더 적은 판관비로 만드는 기업이 낫다는 가설.
SGNA_GP = factor(
    "SGNA_GP",
    F.sgna.ttm() / F.gp.ttm(),
    category="quality",
    label="판관비 / 총이익",
    direction=-1,
    notes="검증 필요",
)

#: 이익의 세금 신호 (Lev & Nissim 2004). 세전이익 대비 세금이 정상 범위를
#: 벗어나면 회계이익의 질을 의심할 근거가 된다.
TAX_EBIT = factor(
    "TAX_EBIT",
    F.taxexp.ttm() / F.ebit.ttm(),
    category="quality",
    label="법인세 / EBIT",
    notes="Lev & Nissim 2004 — 과세소득이 회계이익의 질을 드러낸다",
)
