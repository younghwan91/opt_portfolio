"""
퀄리티 팩터 (36개)

퀀트 관점:
- GP/A (Novy-Marx 2013) 가 미장에서 가장 강건하게 검증된 퀄리티 팩터다.
  순이익은 회계 재량 여지가 크지만 매출총이익은 조작이 어렵다.
- 발생액(AC) 계열은 **낮을수록 좋다** (Sloan 1996). direction=-1 로 명시.
- R&D 집약도는 섹터 편향이 극심해 중립화 없이는 '기술주 베팅'과 같다.
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.dsl.registry import derive_ttm, factor

_SF1 = ("SF1",)
_SF1_SEP = ("SF1", "SEP")

# ------------------------------------------------------------------ 수익성

ROE = factor("ROE", F.netinc / F.equityavg, category="quality", label="ROE", requires=_SF1)
ROA = factor("ROA", F.netinc / F.assetsavg, category="quality", label="ROA", requires=_SF1)
ROE_TTM = derive_ttm(ROE)
ROA_TTM = derive_ttm(ROA)

ROIC = factor("ROIC", (F.ebit - F.taxexp) / F.invcapavg, category="quality",
              label="ROIC", requires=_SF1, notes="NOPAT / 평균 투하자본")
GPIC = factor("GPIC", F.gp / F.invcapavg, category="quality", label="GPIC",
              requires=_SF1, notes="[정의 추정] 매출총이익 / 투하자본")
RIC = factor("RIC", F.rnd / F.invcapavg, category="quality", label="RIC",
             neutralize=("sector",), requires=_SF1,
             notes="[정의 추정] 연구개발비 / 투하자본")

GP_E = factor("GP_E", F.gp / F.equity, category="quality", label="GP/E", requires=_SF1)
GP_A = factor("GP_A", F.gp / F.assets, category="quality", label="GP/A", requires=_SF1,
              notes="Novy-Marx 총이익성 — 미장 최강 퀄리티 팩터")
GP_A_TTM = derive_ttm(GP_A)

GP_IT = factor("GP_IT", F.gp / F.intangibles, category="quality", label="GP/IT",
               neutralize=("sector",), requires=_SF1, notes="[정의 추정] / 무형자산")
OP_IT = factor("OP_IT", F.opinc / F.intangibles, category="quality", label="OP/IT",
               neutralize=("sector",), requires=_SF1, notes="[정의 추정] / 무형자산")
ROIT = factor("ROIT", F.netinc / F.intangibles, category="quality", label="ROIT",
              neutralize=("sector",), requires=_SF1, notes="[정의 추정] / 무형자산")

ROCE = factor("ROCE", F.ebit / (F.assets - F.liabilitiesc), category="quality",
              label="ROCE", requires=_SF1, notes="EBIT / (총자산 − 유동부채)")

# ------------------------------------------------------------- 회전율 · 마진

IT_TURNOVER = factor("IT_TURNOVER", F.revenue / F.intangibles, category="quality",
                     label="무형자산 Turnover", neutralize=("sector",), requires=_SF1)
ASSET_TURNOVER = factor("ASSET_TURNOVER", F.revenue / F.assetsavg, category="quality",
                        label="Asset Turnover", requires=_SF1)
ASSET_TURNOVER_TTM = derive_ttm(ASSET_TURNOVER)

GPM = factor("GPM", F.gp / F.revenue, category="quality", label="GPM", requires=_SF1)
OPM = factor("OPM", F.opinc / F.revenue, category="quality", label="OPM", requires=_SF1)
NPM = factor("NPM", F.netinc / F.revenue, category="quality", label="NPM", requires=_SF1)

# ------------------------------------------------------------------ R&D 집약도

_RND_RATIOS = [
    ("RND_REVENUE", F.revenue, "R&D / 매출액"),
    ("RND_GP", F.gp, "R&D / 매출총이익"),
    ("RND_OPINC", F.opinc, "R&D / 영업이익"),
    ("RND_NETINC", F.netinc, "R&D / 순이익"),
]

RND_FACTORS = {
    name: factor(name, F.rnd / denom, category="quality", label=label,
                 neutralize=("sector",), requires=_SF1,
                 notes="섹터 중립화 없이는 기술주 섹터 베팅과 구분 불가")
    for name, denom, label in _RND_RATIOS
}

# ------------------------------------------------------------- 발생액 (낮을수록 좋음)

_ACCRUAL = F.netinc - F.ncfo

AC_A = factor("AC_A", _ACCRUAL / F.assetsavg, category="quality", label="AC/A",
              direction=-1, requires=_SF1, notes="Sloan(1996) 발생액 이상현상")
AC_E = factor("AC_E", _ACCRUAL / F.equityavg, category="quality", label="AC/E",
              direction=-1, requires=_SF1)

# ------------------------------------------------------------- 안정성 · 재무구조

_DAILY_RET = F.close.pct_change(1)

VOL_52W = factor("VOL_52W", _DAILY_RET.rolling_std(252) * (252 ** 0.5), category="quality",
                 label="변동성 (52주)", direction=-1, requires=("SEP",),
                 notes="저변동성 이상현상 — 낮을수록 좋음")
VOL_60D = factor("VOL_60D", _DAILY_RET.rolling_std(60) * (252 ** 0.5), category="quality",
                 label="변동성 (60일)", direction=-1, requires=("SEP",))

OPINC_DEBT = factor("OPINC_DEBT", F.opinc / F.debt, category="quality",
                    label="영업이익 / 차입금", requires=_SF1, notes="이자보상배율 근사")
DEBT_RATIO = factor("DEBT_RATIO", F.debt / F.equity, category="quality",
                    label="차입금비율", direction=-1, requires=_SF1)
RETENTION = factor("RETENTION", F.retearn / F.equity, category="quality",
                   label="유보율", requires=_SF1)
CURRENT_RATIO = factor("CURRENT_RATIO", F.assetsc / F.liabilitiesc, category="quality",
                       label="유동비율", requires=_SF1)

# 이익변동성: 자산대비 이익률의 20분기 표준편차. 분기 그리드에서 계산하므로
# rolling_std(일별 전용) 대신 표현식 조합이 아니라 전용 노드가 필요하다 →
# 아래 EarningsVolatility 는 quarterly rolling 을 쓰는 별도 구현.
from opt_portfolio.factor.library._quarterly_ops import quarterly_rolling_std  # noqa: E402

EARNINGS_VOL = factor(
    "EARNINGS_VOL",
    quarterly_rolling_std(F.netinc / F.assets, window=20),
    category="quality",
    label="이익변동성",
    direction=-1,
    requires=_SF1,
    notes="ROA 의 20분기 표준편차 — 낮을수록 이익의 예측가능성이 높음",
)

# ------------------------------------------------------------------ 복합 스코어

from opt_portfolio.factor.library._composites import (  # noqa: E402
    altman_z_expr,
    piotroski_f_expr,
)

F_SCORE = factor("F_SCORE", piotroski_f_expr(), category="quality", label="F-score",
                 winsor=0.0, requires=_SF1,
                 notes="Piotroski 9개 이진 항목 합 (0~9). 이산값이라 윈저라이즈 미적용")

ALTMAN_Z = factor("ALTMAN_Z", altman_z_expr(), category="quality", label="Altman Z-score",
                  requires=_SF1_SEP,
                  notes="Z < 1.81 부실 위험. '관리종목 제외' 필터의 미장 대체재. "
                        "금융업에는 부적용 → 금융주 제외 유니버스 전제")
