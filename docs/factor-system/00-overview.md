# US Factor Investing Engine — 설계 개요

미국 주식 전종목(약 6,000종목, 상장폐지 포함)을 대상으로 한
**팩터 계산 → 유니버스 필터 → 멀티팩터 랭킹 → 포트폴리오 최적화 → 백테스트** 시스템.

기존 `opt_portfolio`(ETF 자산배분, VAA/OU)는 **그대로 보존**하고,
종목 단위 횡단면(cross-sectional) 팩터 엔진을 새 서브시스템으로 추가한다.

---

## 1. 왜 새 서브시스템인가

| | 기존 (VAA/OU) | 신규 (Factor Engine) |
|---|---|---|
| 데이터 형태 | 시계열 (자산 ~10개) | **패널** (종목 6,000 × 날짜 6,000 × 팩터 150) |
| 데이터 소스 | yfinance | 유료 API (재무제표 + PIT) |
| 신호 | 시계열 모멘텀 | **횡단면 랭킹** |
| 저장 | DuckDB 캐시 (가격만) | DuckDB **bitemporal 스토어** |
| 최적화 | 자산 5개 비중 그리드서치 | 종목 20~100개, 제약조건 하 QP |

공유되는 것: `analysis/metrics.py`(성과지표), `analysis/risk.py`, `utils/visualization.py`.
재사용하되 신규 코드는 `src/opt_portfolio/factor/` 이하에 격리한다.

---

## 2. 데이터 소스 추천

### 결론: **Sharadar (직판) 를 1차 소스로, FMP 를 보조로**

| 항목 | Sharadar (sharadar.com 직판) | FMP | Polygon |
|---|---|---|---|
| 월 비용 | **$9~29** (Fundamentals/Prices/Investors/Bundle) | $15~99 | $29~199 |
| 재무제표 | SF1, ~150 지표, 1998~ | 30년+ | SEC XBRL 원시 |
| **Point-in-Time** | **`datekey`(공시일) 제공 → PIT 정확** | 약함 (재작성치 덮어씀) | 없음 |
| **상장폐지 종목** | **포함 (16,000+ 티커)** | 별도 엔드포인트 | 제한적 |
| R&D (`rnd`) | **있음** | 있음 | XBRL 파싱 필요 |
| 13F 기관보유 | **SF3 있음** | 있음 | 없음 |
| 내부자 거래 | **SF2 있음** | 있음 | 없음 |
| 코퍼레이트액션 | ACTIONS 테이블 | 있음 | **최상급** |
| 애널리스트 추정치 | **없음** | **있음** | 없음 |

**Sharadar 를 고른 이유 — 팩터 투자에서 PIT 와 상장폐지 종목이 전부이기 때문.**

- 상장폐지 종목이 빠지면 **생존 편향**으로 백테스트 수익률이 구조적으로 부풀려진다.
- 재무 데이터가 재작성치(restated)로 덮어써지면 **미래 정보 누출(look-ahead)** 이 발생한다.
  Sharadar 는 `dimension=ARQ/ART`(as-reported) + `datekey`(실제 공시일) 로 이 둘을 모두 해결한다.
- 게다가 SF2(내부자) + SF3(13F) 가 같은 티커 체계로 붙어 있어 **수급 팩터 프록시**를 바로 만들 수 있다.

> ⚠️ 과거 Nasdaq Data Link 경유 시 $150/mo 였으나, 현재 sharadar.com 직판은 $9~29/mo.
> 구독 전 Bundle 티어에 SF1/SEP/SF2/SF3/ACTIONS/TICKERS 가 모두 포함되는지 확인 필요.

**FMP 를 보조로 두는 이유**: `PEG` 의 정통 정의(주가수익성장비율)는 **선행(forward) EPS 성장률**을 쓴다.
Sharadar 에는 애널리스트 추정치가 없다. 두 갈래 중 선택:

1. FMP Starter($15/mo) 추가 → forward PEG (권장, 총 $44/mo)
2. Sharadar 만 사용 → **trailing PEG** (과거 3~5년 EPS CAGR) 로 대체 — 무료지만 정의가 다름

코드는 두 정의를 `PEG_FWD`, `PEG_TTM` 으로 **둘 다** 제공하고, 소스 없으면 자동 skip 한다.

### 프로바이더 추상화

특정 벤더에 고정하지 않는다. `factor/data/provider.py` 의 `DataProvider` 프로토콜을
Sharadar/FMP 어댑터가 구현하고, 팩터 계산부는 **정규화된 표준 스키마**만 본다.
→ 나중에 벤더를 바꿔도 팩터 코드는 손대지 않는다.

---

## 3. 핵심 설계: 팩터 DSL

### 문제

요청된 팩터는 약 150개다. 각각을 함수로 짜면:
- 코드 6,000줄, TTM/QoQ/YoY 로직이 150번 복붙됨
- 성장 팩터 26개 + 가속 팩터 15개는 **기존 팩터의 단순 변환**인데 중복 구현됨
- PIT 시프트를 한 군데라도 빠뜨리면 조용히 look-ahead 발생

### 해결: 표현식 트리 + 트랜스폼 체이닝

팩터를 **선언**하고, 파생형은 **자동 생성**한다.

```python
# base 팩터 선언 (약 55개만 손으로 작성)
PER  = F.mktcap / F.netinc
GP_A = F.gp / F.assets

# 파생형은 트랜스폼으로 자동 생성 — 손으로 안 짬
PER_TTM      = PER.ttm()            # PER (TTM)
GP_A_YOY     = GP_A.yoy()           # GP/A성장률 (YoY)
NETINC_ACCEL = F.netinc.yoy().accel()   # 순이익성장 가속 (YoY)
```

이 방식으로 커버되는 개수:

| 카테고리 | 요청 개수 | 손으로 작성 | 자동 파생 |
|---|---|---|---|
| 가치 (Price) | 22 | 12 | 10 (TTM 변형) |
| 가치 (EV) | 9 | 9 | 0 |
| 퀄리티 | 36 | 32 | 4 (TTM 변형) |
| 가격 | 26 | 14 | 12 (기간 변형) |
| 성장 | 26 | 0 | **26** |
| 가속 | 15 | 3 | **12** |
| **합계** | **134** | **70** | **64** |

성장·가속 41개가 전부 자동 파생된다. 이게 이 설계의 핵심 이득이다.

### 트랜스폼 목록

| 트랜스폼 | 의미 | 적용 대상 |
|---|---|---|
| `.ttm()` | 직전 4개 분기 합(플로우) / 최신값(스톡) | 손익·현금흐름 |
| `.qoq()` | 전분기 대비 증감률 | 재무 팩터 |
| `.yoy()` | 전년동기 대비 증감률 | 재무 팩터 |
| `.accel()` | 성장률의 차분 (2차 미분) | 성장률 팩터 |
| `.mom(n)` | n개월 모멘텀 | 가격 |
| `.zscore(by=)` | 섹터/전체 표준화 | 전부 |
| `.rank(pct=)` | 백분위 랭킹 | 전부 |
| `.winsor(p)` | 상하위 p% 클리핑 | 전부 |
| `.neutralize(by=)` | 섹터/사이즈 중립화 | 전부 |

**스톡/플로우 구분이 중요하다.** `assets`(스톡)의 TTM 은 합계가 아니라 최신값,
`revenue`(플로우)의 TTM 은 4분기 합계다. 필드 메타데이터에 `kind` 를 박아 자동 처리한다.

---

## 4. Point-in-Time 보장 (구조적 차단)

```
fundamentals 테이블 (bitemporal)
┌──────────┬──────────┬───────────┬──────────┬─────────┐
│ ticker   │ calendar │ datekey   │ dimension│ revenue │
│          │ date     │ (공시일)   │          │         │
├──────────┼──────────┼───────────┼──────────┼─────────┤
│ AAPL     │ 2024-03-31│ 2024-05-02│ ARQ      │ 90753   │
└──────────┴──────────┴───────────┴──────────┴─────────┘
```

모든 팩터 조회는 **반드시** `datekey <= as_of` 로 필터링된다.
`calendardate`(회계기간말) 로 조인하면 결산일과 공시일 사이 최대 90일의
미래 정보가 새어 들어간다 — 이건 백테스트를 조용히 망가뜨리는 가장 흔한 버그다.

구현상 규율에 맡기지 않는다:
- 팩터 엔진은 원시 테이블에 직접 접근할 수 없다
- `PITStore.as_of(date)` 가 반환하는 **스냅샷 뷰**만 볼 수 있다
- 스냅샷은 `datekey <= date` 가 이미 적용된 상태 → 물리적으로 미래를 못 봄

가격 데이터도 동일: 리밸런싱일 `t` 의 신호는 `t` **종가까지**만 쓰고, 체결은 `t+1` 시가.

---

## 5. 수급 팩터의 미국식 대체

요청된 개인/기관/외인 순매수강도는 **KRX 투자자별 매매동향 전용 데이터**로,
미국 시장에는 존재하지 않는다 (미국은 투자자 유형별 일별 거래대금을 공시하지 않음).
아래 프록시로 대체한다.

| 원본 (KRX) | 미국식 프록시 | 데이터 | 발표 주기 | 비고 |
|---|---|---|---|---|
| 기관순매수강도 | **13F 기관보유 주식수 변화율** | Sharadar SF3 | 분기 (45일 지연) | 지연 반영 필수 |
| 기관순매수강도 (보조) | 기관 보유자 **수** 증감 | SF3 | 분기 | 신규 진입 기관 |
| 개인순매수강도 | **내부자(Form 4) 순매수** | Sharadar SF2 | 2영업일 내 | "스마트 개인" |
| 개인순매수강도 (보조) | 1 − 기관보유비율 의 변화 | SF3 + SEP | 분기 | 잔여 지분 근사 |
| 외인순매수강도 | *(직접 대응 없음)* | — | — | 미장은 자국 시장 |
| 외인 대체 | **공매도 잔고(short interest) 변화** | FMP / FINRA | 격주 | 역방향 수급 신호 |
| 기관/외인순매수강도 | 13F 변화 + 공매도 변화 합성 | SF3 + SI | — | |
| 거래대금 회전율 | 그대로 사용 (`volume × close / mktcap`) | SEP | 일별 | 변경 없음 |

> ⚠️ 13F 는 분기말 기준 45일 후 공시된다. 반드시 공시일 기준으로 반영해야 하며,
> 이 지연 때문에 원본 KRX 일별 수급 팩터보다 신호 강도가 현저히 약하다.
> **동일한 팩터로 취급하면 안 되고, 별도 카테고리(`flow_proxy`)로 분리**한다.

---

## 6. 유니버스 (미장 특화)

### 일반 필터

| 필터 | 구현 |
|---|---|
| 금융주 제외 | GICS Sector = Financials 제외 |
| 지주사 제외 | SIC 6719 / 사명 `Holdings?` 패턴 + 매출 대비 지분법이익 비중 |
| 관리종목 제외 | *(미장 미대응)* → **대체**: 상장폐지 경고, `Altman Z < 1.8`, 페니스톡(< $1), 감사의견 거절 |
| 적자기업 제외 (분기) | 최근 분기 `netinc <= 0` 제외 |
| 적자기업 제외 (연간) | TTM `netinc <= 0` 제외 |
| 중국기업 제외 | 본사 소재 CN/HK ADR 제외 (SEC HFCAA 리스트 병행) |
| **PTP 기업 제외** | IRS Sec.1446(f) 대상 PTP — **한국 투자자 매도대금 10% 원천징수** |
| 소형주 하위 20%만 | 시가총액 백분위 하위 20% |

> 💡 PTP 필터는 미장 특유의 실전 항목이다. 2023년부터 비거주 외국인의 PTP 매도 시
> **손익과 무관하게 매도 총액의 10%** 가 원천징수된다. 백테스트에는 안 잡히지만
> 실제 수익률을 파괴하므로 기본값 `exclude_ptp=True` 로 둔다.

### 산업 필터

요청하신 26개 산업 분류는 **한국 WICS 체계**다. 미장에는 GICS 를 쓴다.
WICS→GICS 매핑 테이블을 제공해 기존 사고방식 그대로 쓸 수 있게 한다.

| WICS (한국) | GICS Industry (미국) |
|---|---|
| 건강관리 | Health Care Equipment / Pharma / Biotech |
| 반도체 | Semiconductors & Semiconductor Equipment |
| 소프트웨어 | Software |
| IT하드웨어 | Technology Hardware, Storage & Peripherals |
| 조선 | Machinery — Construction & Farm (근사) |
| 상사,자본재 | Industrial Conglomerates / Trading Companies |
| … | (전체 매핑은 `03-universe-spec.md`) |

**추가로 GICS 원본 선택도 지원한다** — 미장에서는 그쪽이 정확하다.

### 추가 필수 필터 (요청에 없지만 미장에서 반드시 필요)

| 필터 | 이유 |
|---|---|
| **최소 유동성** (20일 평균 거래대금 > $1M) | 없으면 백테스트가 체결 불가능한 종목으로 수익을 만듦 |
| **최소 주가** (> $5) | 페니스톡의 호가 스프레드가 수익률을 왜곡 |
| ADR / 이중상장 중복 제거 | 같은 기업이 두 번 편입되는 것 방지 |
| SPAC / 셸컴퍼니 제외 | 재무비율이 무의미 |

---

## 7. 트레이딩 설정

```yaml
backtest:
  initial_capital: 100_000        # USD (한국 UI 의 "만원" → USD 로 전환)
  commission_pct: 0.05            # 왕복 아님, 편도
  slippage_bps: 10                # 요청에 없었으나 추가 — 없으면 백테스트가 거짓말을 함
  start: 2003-01-01
  end:   2026-08-04

rebalance:
  period: quarterly               # weekly | monthly | quarterly | semiannual | annual
  weighting: equal                # equal | mktcap | inverse_vol | risk_parity | hrp | mvo | black_litterman
  n_stocks: 20
  max_weight: 0.10
  sector_max: 0.30

market_timing:
  momentum: true                  # 지수 이평선 이탈 시 현금화
  macro: false                    # 실업률/장단기금리차/신용스프레드 기반
  reentry: true                   # 청산 후 재진입 조건
```

> ⚠️ **슬리피지는 요청에 없었지만 기본 포함한다.** 수수료 0%로 20종목을 분기 리밸런싱하면
> 백테스트 수익률이 실현 불가능한 수준으로 나온다. 소형주 팩터일수록 심각하다.
> 명시적으로 0 으로 두실 수 있지만, 기본값은 10bp 로 둔다.

### 비중 조절 — 요청은 "동일 비중" 하나였지만

레포의 목표가 **포트폴리오 최적화**이므로, 팩터 스코어를 최적화기에 연결한다:

| 방법 | 설명 |
|---|---|
| `equal` | 동일 비중 (요청 사항, 기본값) |
| `inverse_vol` | 변동성 역가중 |
| `risk_parity` | 리스크 기여도 균등 |
| `hrp` | Hierarchical Risk Parity (공분산 추정오차에 강건) |
| `mvo` | 평균-분산, 팩터 스코어 = 기대수익 |
| **`black_litterman`** | **시장균형 사전분포 + 팩터 스코어를 view 로 주입** ← 창의적 연결점 |

Black-Litterman 이 이 레포에서 특히 맞는 이유: 멀티팩터 스코어는 "이 종목이 좋다"는
**상대적 견해**이지 기대수익률 추정치가 아니다. BL 은 정확히 그 형태의 입력을 받는다.
→ 팩터 엔진과 최적화기가 이론적으로 자연스럽게 접합된다.

### 마켓 타이밍

| 방식 | 규칙 |
|---|---|
| 모멘텀 | S&P500 종가 < 200일 이평 → 전량 현금 (기존 VAA 로직 재사용) |
| 매크로 | 장단기금리차 역전 + 실업률 12개월 이평 상회 → 익스포저 축소 |
| 재진입 | 청산 후 지수가 200일 이평 회복 + N일 유지 → 복귀 (휩쏘 방지) |

---

## 8. 팩터 검증 레이어 (요청에 없었으나 핵심)

150개 팩터를 만들어놓고 검증 없이 쓰면 **과최적화 기계**가 된다. 필수 구성:

| 도구 | 산출물 |
|---|---|
| Rank IC / IC-IR | 팩터별 예측력 시계열, 감쇠(decay) 프로파일 |
| 분위수 스프레드 | 10분위 롱숏 수익률, 단조성(monotonicity) |
| 팩터 상관행렬 | 150개 중 실제 독립적인 건 20~30개 — 중복 제거 |
| 회전율 | 팩터별 리밸런싱 회전율 → 실현 가능성 |
| 섹터/사이즈 중립 IC | 팩터가 진짜인지 섹터 베팅인지 판별 |
| **Deflated Sharpe Ratio** | 150개를 훑은 후 나온 최고 샤프는 거의 확실히 우연 — 다중검정 보정 |

> ⚠️ 150개 팩터 × 여러 파라미터를 탐색하면 순수한 우연으로도 t > 3 이 나온다.
> Deflated Sharpe / PBO(Probability of Backtest Overfitting) 없이는
> 어떤 결과도 신뢰할 수 없다. 이 레이어를 선택이 아닌 **필수 경로**로 둔다.

---

## 9. 디렉터리 구조

```
src/opt_portfolio/
├── (기존 VAA/OU 모듈 — 변경 없음)
│   analysis/  core/  strategies/  ui/  utils/
│
└── factor/                        ← 신규
    ├── data/
    │   ├── provider.py            # DataProvider 프로토콜 (벤더 중립)
    │   ├── sharadar.py            # Sharadar 어댑터
    │   ├── fmp.py                 # FMP 어댑터 (추정치·공매도)
    │   ├── schema.py              # 정규화 표준 스키마 + 필드 메타(stock/flow)
    │   └── store.py               # DuckDB bitemporal PIT 스토어
    ├── dsl/
    │   ├── expr.py                # 표현식 트리 (Expr, BinOp, Field)
    │   ├── transforms.py          # ttm/qoq/yoy/accel/rank/zscore/neutralize
    │   └── registry.py            # @factor 데코레이터, 카테고리, 메타
    ├── library/
    │   ├── value_price.py         # 가치 (Price) 22
    │   ├── value_ev.py            # 가치 (EV) 9
    │   ├── quality.py             # 퀄리티 36
    │   ├── price.py               # 가격 26
    │   ├── growth.py              # 성장 26 (자동 파생)
    │   ├── acceleration.py        # 가속 15 (자동 파생)
    │   └── flow_proxy.py          # 수급 프록시 (13F/내부자/공매도)
    ├── universe/
    │   ├── filters.py             # 일반 필터 (금융/지주/적자/중국/PTP/소형주)
    │   ├── sectors.py             # GICS + WICS↔GICS 매핑
    │   └── builder.py             # PIT 유니버스 구성
    ├── research/
    │   ├── ic.py                  # Rank IC, IC-IR, 감쇠
    │   ├── quantiles.py           # 분위수 스프레드, 단조성, 회전율
    │   ├── correlation.py         # 팩터 상관 / 클러스터링
    │   └── overfitting.py         # Deflated Sharpe, PBO
    ├── portfolio/
    │   ├── score.py               # 멀티팩터 가중 랭킹 합성
    │   ├── weights.py             # equal/invvol/rp/hrp/mvo/BL
    │   └── constraints.py         # 최대비중/섹터한도/회전율 제약
    ├── backtest/
    │   ├── engine.py              # 이벤트드리븐 리밸런싱 백테스터
    │   ├── costs.py               # 수수료 + 슬리피지 + 마켓임팩트
    │   └── timing.py              # 모멘텀/매크로/재진입 마켓타이밍
    └── config.py                  # 유니버스·팩터·트레이딩 설정 스키마
```

---

## 10. 확정이 필요한 항목

아래는 **추정으로 진행하되**, 실제 정의가 다르면 알려주시면 바로 고칩니다.
(퀀터스 등 국내 플랫폼의 약어 정의가 공개 표준이 아니라 추정이 섞여 있습니다.)

| 팩터 | 가정한 정의 | 확신도 |
|---|---|---|
| POR | 시가총액 / 영업이익 | 높음 |
| PCR | 시가총액 / 영업활동현금흐름 | 높음 |
| PFCR | 시가총액 / 잉여현금흐름 | 높음 |
| PGPR | 시가총액 / 매출총이익 | 높음 |
| PRR | 시가총액 / 연구개발비 | 중간 |
| **PAR** | 시가총액 / 총자산 | **중간** |
| **PACR** | 시가총액 / 발생액(accruals) | **낮음** |
| **PITR** | 시가총액 / 무형자산 | **낮음** |
| **GPIC** | 매출총이익 / 투하자본 | 중간 |
| **RIC** | 연구개발비 / 투하자본 | **낮음** |
| GP/IT, OP/IT, ROIT | 각각 / 무형자산 | 중간 |
| AC | 발생액 = 당기순이익 − 영업활동현금흐름 | 높음 |
| 주주수익률 | (배당 + 자사주매입) / 시가총액 | 높음 |
| 개인/기관/외인 순매수강도 | §5 프록시 | — |

---

## 11. 문서 색인

- `01-factor-spec.md` — 150개 팩터 전체 정의 · 계산식 · 소스 필드 매핑
- `02-universe-spec.md` — 유니버스 필터 · WICS↔GICS 전체 매핑
- `03-backtest-spec.md` — 트레이딩 설정 · 비용 모델 · 마켓타이밍
- `04-data-contract.md` — 정규화 스키마 · PIT 규약 · 프로바이더 어댑터
