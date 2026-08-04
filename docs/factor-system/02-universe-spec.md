# 유니버스 스펙 — 필터 · WICS↔섹터 매핑

구현: `factor/universe/filters.py`, `factor/universe/sectors.py`

## 1. 설계 원칙

- **필터는 알파가 아니라 실현 가능성이다.** 유동성·페니스톡 필터를 끄면
  백테스트는 체결 불가능한 종목에서 수익을 만든다.
- **재무 기반 필터는 공시일 기준으로 적용된다.** 회계기간말 기준으로 걸면
  실적 발표 전에 적자를 '미리 알고' 빼는 look-ahead 가 된다.
  `ctx.eval_daily()` 가 이를 보장한다.
- **확인된 것만 제외한다.** 재무 미공시로 지표가 NaN 인 종목은 통과시킨다.
  (예: Altman Z 가 계산 안 되는 종목을 '부실'로 간주하지 않는다)

## 2. 일반 필터 — 요청된 퀀터스 항목의 미장 대응

| 요청 항목 | `UniverseConfig` 필드 | 기본값 | 구현 |
|---|---|---|---|
| 금융주 제외 | `exclude_financials` | **True** | 섹터 ∈ {Financial Services, Real Estate} |
| 지주사 제외 | `exclude_holdings` | False | 사명 `\bHold(ing\|ings)\b` — 미장 의미 약함 |
| 관리종목 제외 | `exclude_distressed` | **True** | **Altman Z < 1.81** (미장 대체재) |
| 적자기업 제외 (분기) | `exclude_deficit_quarter` | False | 최근 분기 `netinc <= 0` |
| 적자기업 제외 (연간) | `exclude_deficit_ttm` | False | TTM `netinc <= 0` |
| 중국기업 제외 | `exclude_china` | **True** | 소재지 `China\|Hong Kong` |
| PTP 기업 제외 | `exclude_ptp` | **True** | 사명 `L.P.\|Partners` + 시드 리스트 |
| 소형주 하위 20%만 | `smallcap_bottom_pct` | None | 시총 백분위 ≤ 지정값 |

### 요청에 없지만 기본 활성인 것

| 필터 | 필드 | 기본값 | 이유 |
|---|---|---|---|
| 최소 주가 | `min_price_usd` | **$5** | 페니스톡 호가 스프레드가 수익률을 왜곡 |
| 최소 유동성 | `min_adv_usd` | **$1M** | 20일 평균 거래대금 — 체결 가능성 |

명시적으로 `0.0` 을 주면 끌 수 있으나, 끄고 나온 백테스트 수치는 실현 불가능하다.

### PTP 필터 주의

IRS Sec.1446(f) 대상 PTP 는 **비거주 외국인 매도 시 손익과 무관하게 매도 총액의
10% 가 원천징수**된다 (2023~). 백테스트에는 안 잡히지만 실수익을 파괴한다.

`PTP_SEED` 의 19개 티커 + 사명 패턴은 **휴리스틱**이다. 정답은 IRS/브로커
공식 리스트이므로, 실전 투입 전 `extra_ptp_tickers` 로 보강할 것.

## 3. 산업 필터 — WICS(한국) → 미국 섹터 매핑

Sharadar TICKERS 는 Morningstar 계열 11개 섹터를 쓴다. 요청된 26개 WICS 분류를
`(섹터, industry 부분일치 키워드)` 쌍으로 매핑해 한국식 사고를 그대로 쓸 수 있게 했다.

```python
UniverseConfig(wics_industries=("반도체", "소프트웨어"))   # 한국식
UniverseConfig(sectors=("Technology", "Healthcare"))      # 미국 원본 (더 정확)
```
둘은 동시 지정 불가 (`__post_init__` 에서 거부).

| WICS | 섹터 | industry 키워드 |
|---|---|---|
| 건강관리 | Healthcare | (전체) |
| 자동차 | Consumer Cyclical | Auto |
| 화장품,의류,완구 | Consumer Cyclical | Apparel, Personal, Leisure |
| 보험 | Financial Services | Insurance |
| 필수소비재 | Consumer Defensive | (전체) |
| 운송 | Industrials | Airlines, Railroads, Trucking, Marine, Logistics |
| 상사,자본재 | Industrials | Conglomerates, Capital, Distribution |
| 비철,목재등 | Basic Materials | Aluminum, Copper, Lumber, Paper, Metals |
| 화학 | Basic Materials | Chemicals |
| 건설,건축관련 | Industrials | Construction, Building, Engineering |
| 에너지 | Energy | (전체) |
| 기계 | Industrials | Machinery, Tools |
| 철강 | Basic Materials | Steel |
| 반도체 | Technology | Semiconductor |
| IT하드웨어 | Technology | Hardware, Computer, Electronic |
| 통신서비스 | Communication Services | Telecom |
| 증권 | Financial Services | Capital Markets, Brokers, Asset Management |
| 디스플레이 | Technology | Display, Optical |
| IT가전 | Technology | Consumer Electronics |
| 소매(유통) | Consumer Cyclical | Retail, Department, Specialty |
| 유틸리티 | Utilities | (전체) |
| 미디어,교육 | Communication Services | Media, Entertainment, Education |
| 은행 | Financial Services | Banks |
| 호텔,레저서비스 | Consumer Cyclical | Lodging, Resorts, Restaurants, Gambling |
| 소프트웨어 | Technology | Software, Information Technology |
| **조선** | Industrials | Marine Shipping, Aerospace — **근사, 미장에 순수 조선업 부재** |

> ⚠️ WICS 는 한국 시장 구조를 반영한 분류라 1:1 대응이 없는 항목이 있다.
> 정밀 분류가 필요하면 `siccode` 를 직접 쓸 것. 미장에서는 `sectors=` 쪽이 정확하다.

## 4. 알려진 한계

| 항목 | 현황 |
|---|---|
| ADR / 이중상장 중복 제거 | **미구현** — 같은 기업이 두 번 편입될 수 있음 |
| SPAC / 셸컴퍼니 제외 | **미구현** — 재무비율이 무의미한 종목이 통과 |
| 과거 시점 지수 구성종목 | 미지원 — Sharadar `sp500` 테이블은 `action=current` 만 제공 |
| 상장폐지 종목 | 데이터 티어에 종속 (무료 = 현재 구성종목뿐 → 생존편향) |

## 5. 검증

`tests/factor/test_production.py::TestUniverse` 가 합성 스토어의 특수 종목
(금융주 FINCO, 중국기업 CHINACO, PTP OILLP, 페니스톡 PENNY, 만년적자 LOSSCO)이
각 필터에 정확히 걸리는지, 그리고 적자 필터가 공시일 기준(PIT)으로 동작하는지
확인한다.
