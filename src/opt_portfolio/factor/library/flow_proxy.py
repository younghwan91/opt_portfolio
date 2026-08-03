"""
수급 팩터의 미국식 프록시 (5개)

요청하신 개인/기관/외인 순매수강도는 **KRX 투자자별 매매동향 전용 데이터**로,
미국 시장에는 존재하지 않는다. 미국은 투자자 유형별 일별 거래대금을 공시하지
않으며, 대신 공개되는 것은 다음 세 가지다:

| 공시 | 주체 | 주기 | 지연 |
|---|---|---|---|
| 13F | 운용자산 $100M 이상 기관 | 분기 | **45일** |
| Form 4 | 내부자(임원·이사·5% 이상 주주) | 수시 | 2영업일 |
| Short Interest | 브로커 집계 (FINRA) | 격주 | 약 8일 |

퀀트 관점:
- **13F 의 45일 지연이 결정적이다.** KRX 일별 수급은 익일 반영이지만 13F 는
  분기말 후 45일이라, 신호가 이미 가격에 반영된 뒤에 도착한다.
  같은 팩터로 취급하면 안 되며, 별도 카테고리(`flow_proxy`)로 분리해
  다른 팩터와 동일 가중으로 섞이지 않게 한다.
- 외인순매수강도에는 직접 대응물이 없다. 미국은 자국 시장이라 '외국인'이
  의미 있는 투자자 분류가 아니다. 역방향 수급 신호인 공매도 잔고 변화로
  대체하되, **같은 것이라고 주장하지 않는다.**
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.dsl.registry import factor

# ------------------------------------------------------- 기관 (13F, SF3)

INST_HOLD_CHG = factor(
    "INST_HOLD_CHG",
    (F.inst_shares / F.sharesbas).qoq(),
    category="flow_proxy",
    label="기관 보유비중 변화 (13F)",
    requires=("SF3", "SF1"),
    notes="기관순매수강도 대체. 분기말+45일 공시 지연 — datekey 기준 반영 필수",
)

INST_HOLDER_CHG = factor(
    "INST_HOLDER_CHG",
    F.inst_holders.qoq(),
    category="flow_proxy",
    label="13F 보고 기관 수 변화",
    requires=("SF3",),
    notes="신규 진입 기관 수. 보유량 변화와 상보적 — 소형주에서 더 유효",
)

# ------------------------------------------------------- 내부자 (Form 4, SF2)

INSIDER_NET_BUY = factor(
    "INSIDER_NET_BUY",
    F.insider_net_shares.ttm() / F.sharesbas,
    category="flow_proxy",
    label="내부자 순매수 강도",
    requires=("SF2", "SF1"),
    notes=(
        "개인순매수강도의 대체재이지만 성격이 다르다. KRX 개인 수급은 "
        "역방향 지표(개인이 사면 하락)로 쓰이는 반면, 내부자 매수는 "
        "정방향 신호다. 부호 방향을 IC 로 반드시 확인할 것."
    ),
)

# ------------------------------------------------------- 공매도 (FINRA / FMP)

SHORT_INT_CHG = factor(
    "SHORT_INT_CHG",
    -((F.short_interest / F.sharesbas).pct_change(21)),
    category="flow_proxy",
    label="공매도 잔고 변화 (역부호)",
    requires=("FMP", "SF1"),
    notes="외인순매수강도의 대체가 아니라 독립적인 역방향 수급 신호. "
          "부호를 반전해 '공매도 감소 = 긍정'으로 정렬",
)

# ------------------------------------------------------------------ 합성

INST_SHORT_COMBO = factor(
    "INST_SHORT_COMBO",
    (F.inst_shares / F.sharesbas).qoq().zscore()
    - (F.short_interest / F.sharesbas).pct_change(21).zscore(),
    category="flow_proxy",
    label="기관/공매도 합성 수급",
    requires=("SF3", "FMP", "SF1"),
    notes="기관/외인순매수강도 슬롯의 대체. 두 신호를 z-score 로 표준화 후 결합",
)

#: 요청 팩터 → 프록시 매핑 (문서화 및 UI 라벨링용)
KRX_PROXY_MAP = {
    "개인순매수강도": "INSIDER_NET_BUY",
    "기관순매수강도": "INST_HOLD_CHG",
    "외인순매수강도": "SHORT_INT_CHG",
    "기관/외인순매수강도": "INST_SHORT_COMBO",
    "거래대금 회전율": "TURNOVER",  # price.py — 미장에서도 그대로 성립
}
