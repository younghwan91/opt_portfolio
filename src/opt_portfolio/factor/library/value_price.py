"""
가치 팩터 — Price 관련 (22개)

퀀트 관점:
- 배수형 팩터(PER/PBR/PSR…)는 `invert=True` 로 등록한다. 배수를 그대로
  오름차순 정렬하면 적자기업의 음수 PER 이 '가장 싼 주식'으로 올라온다.
  역수(수익률 형태)로 스코어링하면 적자기업은 자연스럽게 하위로 간다.
- TTM 변형은 손으로 쓰지 않고 derive_ttm() 으로 생성한다.
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.dsl.registry import derive_ttm, factor

# ------------------------------------------------------------------ 사이즈

SIZE = factor(
    "SIZE",
    F.mcap,
    category="value_price",
    label="시가총액",
    direction=-1,  # 소형주 프리미엄: 작을수록 좋음
    requires=("SEP",),
    notes="Fama-French SMB. 유동성 필터와 함께 써야 실현 가능",
)

# --------------------------------------------------- 이익 · 자산 기반 배수 (12개)

PER = factor("PER", F.mcap / F.netinc, category="value_price", label="PER",
             invert=True, requires=("SF1", "SEP"))
PBR = factor("PBR", F.mcap / F.equity, category="value_price", label="PBR",
             invert=True, requires=("SF1", "SEP"))
PSR = factor("PSR", F.mcap / F.revenue, category="value_price", label="PSR",
             invert=True, requires=("SF1", "SEP"))
POR = factor("POR", F.mcap / F.opinc, category="value_price", label="POR",
             invert=True, requires=("SF1", "SEP"))
PCR = factor("PCR", F.mcap / F.ncfo, category="value_price", label="PCR",
             invert=True, requires=("SF1", "SEP"),
             notes="영업활동현금흐름 기준 — 이익 조정에 덜 취약")
PFCR = factor("PFCR", F.mcap / F.fcf, category="value_price", label="PFCR",
              invert=True, requires=("SF1", "SEP"))
PGPR = factor("PGPR", F.mcap / F.gp, category="value_price", label="PGPR",
              invert=True, requires=("SF1", "SEP"))

PRR = factor("PRR", F.mcap / F.rnd, category="value_price", label="PRR",
             invert=True, neutralize=("sector",), requires=("SF1", "SEP"),
             notes="R&D 미지출 기업이 다수 → NaN. 섹터 중립 필수")
PAR = factor("PAR", F.mcap / F.assets, category="value_price", label="PAR",
             invert=True, requires=("SF1", "SEP"),
             notes="[정의 추정] 시가총액/총자산")
PACR = factor("PACR", F.mcap / (F.netinc - F.ncfo), category="value_price", label="PACR",
              invert=True, requires=("SF1", "SEP"),
              notes="[정의 추정] 시가총액/발생액. 발생액 부호가 불안정해 해석 주의")
PITR = factor("PITR", F.mcap / F.intangibles, category="value_price", label="PITR",
              invert=True, neutralize=("sector",), requires=("SF1", "SEP"),
              notes="[정의 추정] 시가총액/무형자산. 무형자산 0 기업 다수")

# ------------------------------------------------------------- TTM 변형 (자동 생성)

PER_TTM = derive_ttm(PER)
PSR_TTM = derive_ttm(PSR)
POR_TTM = derive_ttm(POR)
PCR_TTM = derive_ttm(PCR)
PGPR_TTM = derive_ttm(PGPR)

# ------------------------------------------------------------------ 그레이엄 NCAV

NCAV = factor(
    "NCAV",
    (F.assetsc - F.liabilities) / F.mcap,
    category="value_price",
    label="NCAV",
    requires=("SF1", "SEP"),
    notes="순유동자산 / 시가총액. >1 이면 청산가치 미만 거래",
)

# ------------------------------------------------------------------ 주주환원

DIV_YIELD = factor(
    "DIV_YIELD",
    (-F.ncfdiv).ttm() / F.mcap,
    category="value_price",
    label="배당수익률",
    requires=("SF1", "SEP"),
    notes="ncfdiv 는 현금유출이라 음수로 기록됨 → 부호 반전",
)

SHAREHOLDER_YIELD = factor(
    "SHAREHOLDER_YIELD",
    ((-F.ncfdiv) + (-F.ncfcommon)).ttm() / F.mcap,
    category="value_price",
    label="주주수익률",
    requires=("SF1", "SEP"),
    notes=(
        "배당 + 자사주매입 순액. 미국 대형주는 배당보다 자사주매입 규모가 커서 "
        "배당수익률만 보면 주주환원을 절반도 못 본다. 증자 시 ncfcommon 이 "
        "양수가 되어 희석이 음의 기여로 잡히는 것도 의도된 동작."
    ),
)

# ------------------------------------------------------------------ PEG (두 정의)

PEG_TTM = factor(
    "PEG_TTM",
    (F.mcap / F.netinc.ttm()) / (F.epsdil.ttm().yoy() * 100.0),
    category="value_price",
    label="PEG (trailing)",
    invert=False,
    direction=-1,  # 낮을수록 좋음
    requires=("SF1", "SEP"),
    notes="Sharadar 단독 계산 가능. 성장률 ≤ 0 이면 NaN",
)

PEG_FWD = factor(
    "PEG_FWD",
    (F.mcap / F.netinc.ttm()) / (F.eps_growth_fwd * 100.0),
    category="value_price",
    label="PEG (forward)",
    direction=-1,
    requires=("SF1", "SEP", "FMP"),
    notes="정통 정의. FMP 애널리스트 추정치 필요 — 미구독 시 자동 비활성화",
)
