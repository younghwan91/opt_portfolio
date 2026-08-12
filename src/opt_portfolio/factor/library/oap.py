"""
Open Asset Pricing 복제 — Chen & Zimmermann 라이브러리에서 옮긴 팩터.

출처: github.com/OpenSourceAP/CrossSection 의 `SignalDoc.csv`.
이 라이브러리는 원 논문의 방법을 그대로 재현해 t값을 공개하며, 우리는
그중 **회계 기반 · 연간 리밸런싱(저회전) · |t| ≥ 3** 인 것만 가져왔다.

왜 이 목록인가:
- Hou·Xue·Zhang(2020)은 452개 아노말리의 65%가 t>1.96 을 못 넘는다고 보고하지만,
  Chen·Zimmermann 은 그 452개가 실제로는 240개 특성이고 그중 원 논문에서
  **명확히 유의했던 118개 가운데 117개가 재현된다**고 반박한다. HXZ 의 실패율은
  상당 부분 애초에 유의하지 않았던 것들의 오분류다.
- 다만 C&Z 의 재현은 대부분 **균등가중·마이크로캡 포함** 조건이다. 우리 유니버스
  (대형·중형, 비용 20~25bp)에서는 문헌 수익의 **약 1/3 이 깎인다** — C&Z 자신이
  월 60bp → 40bp 로 측정했고, S&P500 급으로 좁히면 그보다 더 깎인다.
  그래서 여기 t값은 채택 근거가 아니라 **후보 선별 근거**일 뿐이다.
  채택은 우리 10분할 테스트와 비용 반영 walk-forward 로만 한다.

Compustat → Sharadar 필드 대응:
    at→assets · act→assetsc · che→cashneq · lt→liabilities · lct→liabilitiesc
    dlc→debtc · dltt→debtnc · ceq/seq→equity · invt→inventory · capx→capex
`ivao`(비유동 투자자산)는 우리 스키마에 없어 0 으로 둔다 — 대부분 기업에서 작다.
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import Expr, F
from opt_portfolio.factor.dsl.registry import factor

_SEP = ("SF1", "SEP")
_YEAR = 4  # 분기 그리드에서 1년


def _avg_assets() -> Expr:
    """전년·당년 평균 총자산 — Richardson 계열의 표준 스케일러."""
    return (F.assets + F.assets.lag(_YEAR)) / 2.0


def _delta(expr: Expr, periods: int = _YEAR) -> Expr:
    return expr - expr.lag(periods)


# ------------------------------------------------------- Richardson et al. 2005
#: 유동영업자산 증가 (t=8.71, JAE). 운전자본이 부푸는 기업의 이익은 덜 지속된다.
DEL_COA = factor(
    "DEL_COA",
    _delta(F.assetsc - F.cashneq) / _avg_assets(),
    category="quality",
    label="유동영업자산 증가",
    direction=-1,
    notes="Richardson et al. 2005 JAE, OAP t=8.71",
)

#: 금융부채 증가 (t=8.01, JAE). 외부 차입으로 자산을 늘린 기업의 이후 수익이 낮다.
DEL_FINL = factor(
    "DEL_FINL",
    _delta(F.debtnc + F.debtc) / _avg_assets(),
    category="quality",
    label="금융부채 증가",
    direction=-1,
    notes="Richardson et al. 2005 JAE, OAP t=8.01",
)

#: 자기자본 증가 (t=6.25, JAE). 증자·유보 어느 쪽이든 자본이 불면 수익률이 낮다.
DEL_EQU = factor(
    "DEL_EQU",
    _delta(F.equity) / _avg_assets(),
    category="quality",
    label="자기자본 증가",
    direction=-1,
    notes="Richardson et al. 2005 JAE, OAP t=6.25",
)

# ------------------------------------------------------------- Soliman 2008 AR
#: 순비유동영업자산 변화 (t=5.26, AR). DuPont 분해의 자산 측 신호.
CH_NNCOA = factor(
    "CH_NNCOA",
    _delta(((F.assets - F.assetsc) - (F.liabilities - F.debtc - F.debtnc)) / F.assets),
    category="quality",
    label="순비유동영업자산 변화",
    direction=-1,
    notes="Soliman 2008 AR, OAP t=5.26 (ivao 미보유 → 0 처리)",
)

#: 순운전자본 변화 (t=4.61, AR).
CH_NWC = factor(
    "CH_NWC",
    _delta(((F.assetsc - F.cashneq) - (F.liabilitiesc - F.debtc)) / F.assets),
    category="quality",
    label="순운전자본 변화",
    direction=-1,
    notes="Soliman 2008 AR, OAP t=4.61",
)

#: 자산회전율 변화 (t=5.12, AR). DuPont 분해에서 마진보다 회전율 변화가
#: 이익의 지속성을 더 잘 예측한다는 것이 이 논문의 핵심이다.
CH_ASSET_TURNOVER = factor(
    "CH_ASSET_TURNOVER",
    _delta(F.revenue.ttm() / F.assets),
    category="quality",
    label="자산회전율 변화",
    notes="Soliman 2008 AR, OAP t=5.12",
)

# ------------------------------------------------------------------ 자금조달·투자
#: 복합 부채발행 (t=8.59, RFS). 5년에 걸친 장부부채의 로그 증가율.
#: 단년 발행보다 저주파라 분기 리밸런싱과 궁합이 좋다.
COMPOSITE_DEBT_ISSUANCE = factor(
    "COMPOSITE_DEBT_ISSUANCE",
    (F.debt / F.debt.lag(5 * _YEAR)),
    category="quality",
    label="복합 부채발행 (5년)",
    direction=-1,
    notes="Lyandres, Sun & Zhang 2008 RFS, OAP t=8.59",
)

#: 재고 증가 (t=6.64, RFS). 실물투자 과잉의 직접 신호.
INV_GROWTH = factor(
    "INV_GROWTH",
    _delta(F.inventory) / _avg_assets(),
    category="quality",
    label="재고 증가",
    direction=-1,
    notes="Belo & Lin 2012 RFS, OAP t=6.64",
)

#: 설비투자 증가율 (t=5.05, JF). 수준(강도)이 아니라 **증가율**이라는 점에서
#: 기존 CAPEX_INTENSITY 와 다르다 — 우리 10분할에서 수준 버전은 실패했다.
GR_CAPX = factor(
    "GR_CAPX",
    (-F.capex).ttm() / (-F.capex).ttm().lag(2 * _YEAR),
    category="quality",
    label="설비투자 증가율 (2년)",
    direction=-1,
    neutralize=("sector",),
    notes="Anderson & Garcia-Feijoo 2006 JF, OAP t=5.05",
)

#: 자기자본 성장 (t=5.38, JFR).
CH_EQUITY = factor(
    "CH_EQUITY",
    F.equity / F.equity.lag(_YEAR),
    category="quality",
    label="자기자본 성장",
    direction=-1,
    notes="Lockwood & Prombutr 2010 JFR, OAP t=5.38",
)

# ------------------------------------------------------------- Fama-French 1992
#: 자산/시가총액 (t=5.69, JF). PBR 이 자기자본을 쓰는 것과 달리 **총자산**을
#: 쓴다 — 레버리지가 다른 기업 간 비교에서 장부가치를 다르게 포착한다.
ASSETS_TO_MARKET = factor(
    "ASSETS_TO_MARKET",
    F.assets / F.mcap,
    category="value_price",
    label="자산 / 시가총액",
    requires=_SEP,
    notes="Fama & French 1992 JF, OAP t=5.69",
)

#: 장부 레버리지 (t=5.34, JF).
BOOK_LEVERAGE = factor(
    "BOOK_LEVERAGE",
    F.assets / F.equity,
    category="quality",
    label="장부 레버리지",
    direction=-1,
    notes="Fama & French 1992 JF, OAP t=5.34",
)
