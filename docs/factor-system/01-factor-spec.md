# 팩터 전체 정의 스펙

소스 필드명은 **Sharadar SF1/SEP/SF2/SF3** 기준.
`schema.py` 의 정규화 레이어를 거치므로 다른 벤더로 바꿔도 팩터 정의는 불변.

## 표기 규약

| 기호 | 의미 |
|---|---|
| `mcap` | 시가총액 (`marketcap`, 일별 = `close × sharesbas × sharefactor`) |
| `⟨x⟩ₜₜₘ` | 직전 4개 분기 합 (플로우) |
| `⟨x⟩ₐᵥg` | 기초·기말 평균 (스톡, ROE/ROA 분모용) |
| `AC` | 발생액 = `netinc − ncfo` |
| `IC` | 투하자본 (`invcap`) |
| `IT` | 무형자산 (`intangibles`) |
| **[auto]** | DSL 트랜스폼으로 자동 생성 (손으로 작성 안 함) |
| **[?]** | 정의 추정 — `00-overview.md` §10 참조 |

> **주의**: 밸류 팩터는 저PER 이 좋으므로 랭킹 시 **역수(earnings yield 형태)를 쓴다.**
> `PER` 을 그대로 오름차순 정렬하면 적자기업(음수 PER)이 최상위로 올라온다.
> 엔진은 `mcap/netinc` 가 아니라 `netinc/mcap` 을 내부 스코어로 쓰고,
> 표시용으로만 역수를 보여준다. (`invert=True` 메타)

---

## 1. 가치 팩터 — Price 관련 (22)

| # | 팩터 | 계산식 | 소스 필드 | 비고 |
|---|---|---|---|---|
| 1 | 시가총액 | `mcap` | `marketcap` | 사이즈 팩터 |
| 2 | PER | `mcap / netinc` | `netinc` | 분기 연율화 |
| 3 | PER (TTM) | `mcap / ⟨netinc⟩ₜₜₘ` | | **[auto]** |
| 4 | PBR | `mcap / equity` | `equity` | 스톡 |
| 5 | PSR | `mcap / revenue` | `revenue` | |
| 6 | PSR (TTM) | `mcap / ⟨revenue⟩ₜₜₘ` | | **[auto]** |
| 7 | POR | `mcap / opinc` | `opinc` | |
| 8 | POR (TTM) | `mcap / ⟨opinc⟩ₜₜₘ` | | **[auto]** |
| 9 | PCR | `mcap / ncfo` | `ncfo` | 영업활동현금흐름 |
| 10 | PCR (TTM) | `mcap / ⟨ncfo⟩ₜₜₘ` | | **[auto]** |
| 11 | PFCR | `mcap / fcf` | `fcf` = `ncfo − capex` | |
| 12 | PRR | `mcap / rnd` | `rnd` | R&D 0 인 기업 다수 → NaN 처리 **[?]** |
| 13 | PGPR | `mcap / gp` | `gp` | |
| 14 | PGPR (TTM) | `mcap / ⟨gp⟩ₜₜₘ` | | **[auto]** |
| 15 | PEG | `PER_TTM / EPS성장률` | `epsdil` | 아래 참조 |
| 16 | PAR | `mcap / assets` | `assets` | **[?]** |
| 17 | PACR | `mcap / AC` | `netinc`,`ncfo` | **[?]** 부호 불안정 |
| 18 | PITR | `mcap / intangibles` | `intangibles` | **[?]** 무형자산 0 다수 |
| 19 | NCAV | `(assetsc − liabilities) / mcap` | `assetsc`,`liabilities` | 그레이엄 순유동자산 |
| 20 | 배당수익률 | `dps_ttm / close` | `dps`, `divyield` | |
| 21 | 주주수익률 | `(배당 + 자사주매입) / mcap` | `−ncfdiv`, `−ncfcommon` | 아래 참조 |
| 22 | 종가 (수정 전) | `closeunadj` | SEP `closeunadj` | 팩터 아님, 필터용 |

### PEG 두 정의
```
PEG_FWD = PER_TTM / (선행 EPS 성장률 × 100)     # FMP 추정치 필요
PEG_TTM = PER_TTM / (3년 EPS CAGR × 100)        # Sharadar 단독 가능
```
성장률 ≤ 0 이면 NaN. 두 팩터를 **별도 등록**하고 소스 없으면 자동 skip.

### 주주수익률
```
shareholder_yield = (dividends_paid + net_buyback) / mcap
  dividends_paid = −ncfdiv          # 현금유출이라 음수로 기록됨
  net_buyback    = −ncfcommon       # 자사주매입 순액 (증자 시 음수 → 희석)
```
자사주매입을 무시하면 미장에서 치명적이다. 미국 대형주는 배당보다 **자사주매입 규모가 크다**
(AAPL 등). 한국식 "배당수익률"만 보면 미장 주주환원을 절반도 못 본다.

---

## 2. 가치 팩터 — EV 관련 (9)

`EV = mcap + 총차입금 + 소수주주지분 − 현금성자산`
Sharadar 는 `ev` 를 직접 제공하나, 정합성을 위해 자체 계산도 지원한다.
```
EV = marketcap + debtusd − cashnequsd
```

| # | 팩터 | 계산식 | 소스 필드 |
|---|---|---|---|
| 23 | EV | `ev` | `ev` |
| 24 | EV/Net | `ev / netinc` | `netinc` |
| 25 | EV/Sales | `ev / revenue` | `revenue` |
| 26 | EV/EBITDA | `ev / ebitda` | `evebitda` (사전계산 존재) |
| 27 | EV/EBIT | `ev / ebit` | `evebit` (사전계산 존재) |
| 28 | EV/GP | `ev / gp` | `gp` |
| 29 | EV/R&D | `ev / rnd` | `rnd` |
| 30 | EV/CF | `ev / ncfo` | `ncfo` |
| 31 | EV/AC | `ev / AC` | `netinc`,`ncfo` |

> EV 배수는 자본구조 중립이라 **금융주 제외 유니버스에서 PER/PBR 보다 안정적**이다.
> 다만 순현금 기업은 EV 가 음수가 되어 배수 부호가 뒤집힌다 → `EV ≤ 0` 은 NaN 처리.

---

## 3. 퀄리티 팩터 (36)

### 3.1 수익성

| # | 팩터 | 계산식 | 소스 필드 |
|---|---|---|---|
| 32 | ROE | `netinc / ⟨equity⟩ₐᵥg` | `roe` |
| 33 | ROA | `netinc / ⟨assets⟩ₐᵥg` | `roa` |
| 34 | ROE (TTM) | `⟨netinc⟩ₜₜₘ / ⟨equity⟩ₐᵥg` | **[auto]** |
| 35 | ROA (TTM) | `⟨netinc⟩ₜₜₘ / ⟨assets⟩ₐᵥg` | **[auto]** |
| 36 | ROIC | `NOPAT / ⟨invcap⟩ₐᵥg` | `roic` |
| 37 | GPIC | `gp / ⟨invcap⟩ₐᵥg` | `gp`,`invcapavg` **[?]** |
| 38 | RIC | `rnd / ⟨invcap⟩ₐᵥg` | `rnd`,`invcapavg` **[?]** |
| 39 | GP/E | `gp / equity` | `gp`,`equity` |
| 40 | GP/A | `gp / assets` | Novy-Marx 총이익성 |
| 41 | GP/A (TTM) | `⟨gp⟩ₜₜₘ / assets` | **[auto]** |
| 42 | GP/IT | `gp / intangibles` | **[?]** |
| 43 | OP/IT | `opinc / intangibles` | **[?]** |
| 44 | ROIT | `netinc / intangibles` | **[?]** |
| 45 | ROCE | `ebit / (assets − liabilitiesc)` | `ebit`,`assets`,`liabilitiesc` |

> **GP/A 는 미장에서 가장 잘 검증된 퀄리티 팩터다** (Novy-Marx 2013).
> 순이익은 회계 재량이 많이 개입되지만 매출총이익은 상대적으로 조작이 어렵다.

### 3.2 회전율 · 마진

| # | 팩터 | 계산식 |
|---|---|---|
| 46 | 무형자산 Turnover | `revenue / intangibles` |
| 47 | Asset Turnover | `revenue / ⟨assets⟩ₐᵥg` (`assetturnover`) |
| 48 | Asset Turnover (TTM) | `⟨revenue⟩ₜₜₘ / ⟨assets⟩ₐᵥg` **[auto]** |
| 49 | GPM | `gp / revenue` (`grossmargin`) |
| 50 | OPM | `opinc / revenue` |
| 51 | NPM | `netinc / revenue` (`netmargin`) |

### 3.3 R&D 집약도

| # | 팩터 | 계산식 |
|---|---|---|
| 52 | R&D / 매출액 | `rnd / revenue` |
| 53 | R&D / 매출총이익 | `rnd / gp` |
| 54 | R&D / 영업이익 | `rnd / opinc` |
| 55 | R&D / 순이익 | `rnd / netinc` |

> R&D 는 **섹터 편향이 극심하다** (제약·반도체·소프트웨어에 집중).
> 반드시 `.neutralize(by="sector")` 를 적용해야 의미 있는 신호가 된다.
> 미적용 시 이 팩터는 사실상 "기술주 섹터 베팅"이다.

### 3.4 발생액 (Accruals)

| # | 팩터 | 계산식 | 방향 |
|---|---|---|---|
| 56 | AC/A | `(netinc − ncfo) / ⟨assets⟩ₐᵥg` | **낮을수록 좋음** |
| 57 | AC/E | `(netinc − ncfo) / ⟨equity⟩ₐᵥg` | **낮을수록 좋음** |

> Sloan(1996) 발생액 이상현상. 이익의 질을 나타내며 **부호 방향이 반대**다.
> 레지스트리에 `direction=-1` 로 명시한다.

### 3.5 안정성 · 재무구조

| # | 팩터 | 계산식 | 소스 |
|---|---|---|---|
| 58 | 변동성 (52주) | `std(일간수익률, 252) × √252` | SEP |
| 59 | 변동성 (60일) | `std(일간수익률, 60) × √252` | SEP |
| 60 | 영업이익 / 차입금 | `opinc / debt` | `opinc`,`debt` |
| 61 | 차입금비율 | `debt / equity` (`de`) | |
| 62 | 유보율 | `retearn / equity` | `retearn` |
| 63 | 이익변동성 | `std(⟨netinc/assets⟩, 최근 20분기)` | 낮을수록 좋음 |
| 64 | 유동비율 | `assetsc / liabilitiesc` (`currentratio`) | |

### 3.6 복합 스코어

| # | 팩터 | 구성 |
|---|---|---|
| 65 | **F-score** | Piotroski 9개 이진 항목 합 (0~9) |
| 66 | **Altman Z-score** | 5개 항목 가중합, 부실 예측 |

**Piotroski F-score (9점)**
| 범주 | 조건 |
|---|---|
| 수익성 | ① ROA > 0 ② CFO > 0 ③ ROA 전년 대비 증가 ④ CFO > 순이익 (발생액 품질) |
| 레버리지/유동성 | ⑤ 장기차입금비율 감소 ⑥ 유동비율 증가 ⑦ 신주발행 없음 |
| 운영효율 | ⑧ 매출총이익률 증가 ⑨ 자산회전율 증가 |

**Altman Z-score (제조업)**
```
Z = 1.2·(운전자본/자산) + 1.4·(유보이익/자산) + 3.3·(EBIT/자산)
  + 0.6·(시가총액/총부채) + 1.0·(매출/자산)
```
| 구간 | 해석 |
|---|---|
| Z > 2.99 | 안전 |
| 1.81 ~ 2.99 | 회색지대 |
| Z < 1.81 | 부실 위험 → **"관리종목 제외" 필터의 미장 대체재** |

> 비제조업·금융업은 Z'' 변형 공식을 써야 한다. 금융주 제외 유니버스가 기본인 이유.

---

## 4. 가격 팩터 (26)

### 4.1 모멘텀

| # | 팩터 | 계산식 | 비고 |
|---|---|---|---|
| 67 | 1개월 모멘텀 | `close/close₍₋₂₁₎ − 1` | **단기 반전** → `direction=-1` 권장 |
| 68 | 3개월 모멘텀 | `close/close₍₋₆₃₎ − 1` | **[auto]** |
| 69 | 6개월 모멘텀 | `close/close₍₋₁₂₆₎ − 1` | **[auto]** |
| 70 | 12개월 모멘텀 | `close/close₍₋₂₅₂₎ − 1` | **[auto]** |

> 정통 모멘텀은 **12-1 (직전 1개월 제외)** 을 쓴다. 최근 1개월은 반전 효과가 지배해서
> 그냥 12개월을 쓰면 두 효과가 상쇄된다. `MOM_12_1` 을 추가로 등록한다.

| # | 팩터 | 계산식 |
|---|---|---|
| 71 | 종가 (수정 전) | `closeunadj` — 필터용 |

### 4.2 위험조정 모멘텀

| # | 팩터 | 계산식 |
|---|---|---|
| 72 | 샤프비율 | `mean(r) / std(r) × √252` (252일) |
| 73–76 | 샤프비율 모멘텀 (20/60/120/200일) | 해당 기간 샤프의 변화 **[auto]** |
| 77 | Sortino 비율 | `mean(r) / std(r⁻) × √252` |
| 78–81 | Sortino 모멘텀 (20/60/120/200일) | **[auto]** |

### 4.3 오실레이터

| # | 팩터 | 계산식 |
|---|---|---|
| 82 | RSI (9일) | Wilder RSI |
| 83 | RSI (14일) | **[auto]** |
| 84 | RSI (25일) | **[auto]** |

### 4.4 베타

| # | 팩터 | 계산식 |
|---|---|---|
| 85 | 베타 | `cov(rᵢ, r_mkt)/var(r_mkt)`, 252일 |
| 86 | 베타 (60일) | 60일 **[auto]** |
| 87 | 절대값 베타 | `\|beta\|` — 시장중립 종목 선별 |
| 88 | 절대값 베타 (60일) | **[auto]** |

> 벤치마크는 **SPY 가 아니라 CRSP 시총가중 전체시장** 이 이론적으로 맞다.
> 실무상 SPY 로 근사하되 설정 가능하게 둔다 (`benchmark: SPY`).

### 4.5 수급 프록시 (미국식 대체)

| # | 원본 | 프록시 팩터 | 계산식 | 소스 |
|---|---|---|---|---|
| 89 | 개인순매수강도 | `INSIDER_NET_BUY` | 90일 내부자 순매수 주식수 / 유통주식수 | SF2 |
| 90 | 기관순매수강도 | `INST_HOLD_CHG` | 13F 보유주식수 QoQ 변화율 | SF3 |
| 91 | 외인순매수강도 | `SHORT_INT_CHG` | 공매도 잔고 변화율 (**역방향**) | FMP/FINRA |
| 92 | 기관/외인순매수강도 | `INST_SHORT_COMBO` | `z(INST_HOLD_CHG) − z(SHORT_INT_CHG)` | 합성 |
| 93 | 거래대금 회전율 | `TURNOVER` | `⟨volume × close⟩₂₀ / mcap` | SEP |

> **13F 지연 처리 필수**: 분기말 + 45일 공시. `datekey` 기준으로만 반영한다.
> 이 지연 때문에 KRX 일별 수급 대비 신호가 훨씬 약하며, 별도 카테고리로 분리해
> 다른 팩터와 동일 가중으로 섞이지 않게 한다.

---

## 5. 성장 팩터 (26) — 전부 [auto]

`base.qoq()` / `base.yoy()` 로 **자동 생성**. 손으로 작성하는 코드 0줄.

| base 필드 | QoQ | YoY |
|---|---|---|
| `netinc` (순이익) | ✔ | ✔ |
| `opinc` (영업이익) | ✔ | ✔ |
| `gp` (매출총이익) | ✔ | ✔ |
| `revenue` (매출액) | ✔ | ✔ |
| `assets` (자산) | ✔ | ✔ |
| `equity` (자본) | ✔ | ✔ |
| `GP_A` (GP/A) | ✔ | ✔ |
| `OPINC_DEBT` (영업이익/차입금) | ✔ | ✔ |
| `ncfo` (현금흐름) | ✔ | ✔ |
| `rnd` (연구개발비) | ✔ | ✔ |
| `cashneq` (현금성자산) | ✔ | ✔ |
| `debt` (차입금) | ✔ | ✔ |
| `divyield` (배당수익률) | ✔ | ✔ |

= 13 × 2 = **26개**

### 성장률 계산의 함정

```python
growth = (x_t - x_{t-4}) / abs(x_{t-4})     # abs() 필수
```
`abs()` 없이 계산하면 **적자→적자축소** 기업의 성장률 부호가 뒤집힌다.
(예: `-100 → -50` 은 개선인데, `(-50 - -100)/-100 = -0.5` 로 악화로 잡힘)

추가로 `x_{t-4}` 가 0 근처면 성장률이 폭발한다 →
분모 절대값이 **매출액의 0.1% 미만**이면 NaN 처리하는 가드를 둔다.

---

## 6. 가속 팩터 (15)

### 6.1 이동평균 모멘텀 가속 (3) — 직접 구현

`(a/b/c)` = 단기 이평 `a`개월, 장기 이평 `b`개월, 관찰 `c`일 로 해석한다. **[?]**

| # | 팩터 | 계산식 |
|---|---|---|
| 94 | MA 모멘텀 가속 (3/3/10) | `Δ₁₀[ MA₃(close)/MA₃(close)₍₋ₙ₎ ]` |
| 95 | MA 모멘텀 가속 (3/12/5) | `Δ₅[ MA₃(close)/MA₁₂(close) ]` |
| 96 | MA 모멘텀 가속 (10/1/5) | `Δ₅[ MA₁₀(close)/MA₁(close) ]` |

> 파라미터 해석이 불확실합니다. `(단기/장기/관찰)` 로 가정했으며,
> 실제 정의를 알려주시면 즉시 수정합니다. 구현은 파라미터화해두어
> `ma_accel(short, long, lookback)` 한 함수로 어떤 조합이든 생성 가능합니다.

### 6.2 재무 성장 가속 (12) — 전부 [auto]

`base.yoy().accel()` = 성장률의 차분 = **2차 미분**.

| base | YoY 가속 | QoQ 가속 |
|---|---|---|
| `netinc` (순이익) | ✔ | ✔ |
| `revenue` (매출액) | ✔ | ✔ |
| `gp` (매출총이익) | ✔ | ✔ |
| `opinc` (영업이익) | ✔ | ✔ |
| `ncfo` (영업활동현금흐름) | ✔ | ✔ |
| `AC` (발생액) | ✔ | ✔ |

= 6 × 2 = **12개**

> 가속 팩터는 **노이즈가 매우 크다** (2차 차분 = 노이즈 2번 증폭).
> 단독 사용보다 성장 팩터와의 결합, 또는 `.winsor(0.01)` 후 사용을 권장한다.
> IC 검증 단계에서 대부분 탈락할 가능성이 높으며, 그걸 확인하는 게 검증 레이어의 역할이다.

---

## 7. 집계

| 카테고리 | 개수 | 손으로 작성 | 자동 파생 |
|---|---|---|---|
| 가치 (Price) | 22 | 12 | 10 |
| 가치 (EV) | 9 | 9 | 0 |
| 퀄리티 | 36 | 32 | 4 |
| 가격 | 27 | 15 | 12 |
| 성장 | 26 | 0 | **26** |
| 가속 | 15 | 3 | **12** |
| **합계** | **135** | **71** | **64** |

**64개(47%)가 DSL 트랜스폼으로 자동 생성**된다.

---

## 8. 팩터 메타데이터

각 팩터는 아래 메타를 갖는다 — 검증·최적화 단계에서 사용:

```python
@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: Literal["value_price","value_ev","quality","price",
                      "growth","acceleration","flow_proxy"]
    expr: Expr                    # 표현식 트리
    direction: int = +1           # +1 = 클수록 좋음, -1 = 작을수록 좋음
    invert: bool = False          # 배수형(PER 등) → 역수로 스코어링
    neutralize: tuple = ()        # ("sector",) / ("sector","size")
    winsor: float = 0.01
    min_periods: int = 4          # 필요한 최소 분기 수
    requires: frozenset = ...      # 필요 소스 테이블 {"SF1","SEP","SF3"}
    notes: str = ""
```

`requires` 로 **구독하지 않은 데이터셋에 의존하는 팩터는 자동 비활성화**된다.
(예: FMP 미구독 시 `PEG_FWD`, `SHORT_INT_CHG` 자동 skip — 에러 대신 경고)
