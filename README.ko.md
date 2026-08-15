# opt_portfolio

**한국어** · [English](README.md)

**미국 주식 팩터 투자 엔진 + ETF 전술적 자산배분(VAA) 백테스트 시스템.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

두 개의 독립된 서브시스템이 한 저장소에 있다.

| | **팩터 엔진** (`factor/`) | **VAA 자산배분** (`strategies/`·`analysis/`) |
|---|---|---|
| 대상 | 미국 개별주 (21,963종목, 1997~2026) | ETF 7~11종 |
| 질문 | 어떤 종목을 살까 | 어떤 자산군으로 갈아탈까 |
| 데이터 | Sharadar 직판 (PIT · 상장폐지 포함) | yfinance 일간 종가 |
| 진입점 | `opt-factor` · `opt-factor-tui` | `make run` · `run.py` |

**두 시스템은 코드 수준에서 완전히 격리돼 있다** — 서로를 import 하지 않으며, 공유하는 것은 `config.RISK_FREE_RATE` 하나뿐이다.

---

# 1. 팩터 엔진

미국 주식 횡단면 팩터 전략을 **검증 가능한 방식으로** 만드는 엔진이다.
핵심 설계 원칙은 하나다 — **조용히 틀리지 않는다.**

## 왜 이 엔진인가

퀀트 백테스트가 실패하는 방식은 대개 정해져 있고, 이 엔진은 그 각각을 구조로 막는다.

| 흔한 실패 | 이 엔진의 대응 |
|---|---|
| 생존편향 — 지금 살아남은 종목만 본다 | 상장폐지 종목 포함 (엔론·구 아메리칸항공 등 실제 확인) |
| Look-ahead — 발표 전 숫자를 쓴다 | 표현식이 원시 테이블에 접근 못 하고 `PanelContext` 만 통과, `datekey` 정렬 강제 |
| 재공시 오염 — 정정된 숫자를 쓴다 | **최초 공시 우선** — 시장이 처음 본 값만 저장 |
| 조용한 절단 — 데이터가 덜 왔는데 성공 처리 | 페이지네이션이 기대 범위 미달 시 `TruncatedDataError` |
| 과최적화 — 수백 번 돌려 최고를 고른다 | **DSR**(Deflated Sharpe) + **PBO** 로 시도 횟수를 정산 |
| 인샘플 성과 보고 | 공식 성과는 **walk-forward** 뿐. 단일 백테스트는 참고용 명시 |

## 성과

<!-- PERFORMANCE:START -->

*채택 전략 · walk-forward 검증 구간 · 2007-12 – 2026-08 (18.6년)*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/performance-dark.png">
  <img alt="walk-forward 검증 성과 — SPY 대비" src="docs/images/performance-light.png">
</picture>

| 지표 | 전략 | SPY (같은 구간) |
|---|---|---|
| 연평균 수익률 | **16.90%** | 11.36% |
| 최대낙폭 | **-23.7%** | -52.3% |
| 변동성 | **15.7%** | 19.8% |
| Sharpe | **0.756** | 0.390 |
| Calmar | **0.71** | 0.22 |
| **Deflated Sharpe** (시도 57회) | **0.992** ✓ | — |

검증 구간 누적 18.2배. **보유 종목은 공개하지 않는다** — 초소형주라 공개 추천이 몰리면 자신의 체결가가 나빠진다.

<!-- PERFORMANCE:END -->

채택 전략: 미국 소형주, 8팩터 + 200일 이평 타이밍, 20종목, 분기 리밸런싱, 동일비중.

**Deflated Sharpe 가 이 표에서 가장 중요한 숫자다.** 같은 데이터로 여러 번 시도해 고른 결과에서, 순수한 노이즈만으로 기대되는 최대 Sharpe 를 빼고 남는 값이다. 0.95 를 넘지 못하면 채택하지 않는다 — 실제로 이 저장소에서 20개 넘는 전략이 이 관문에서 탈락했다.

전 과정 기록은 [`docs/factor-system/07-experiment-log.md`](docs/factor-system/07-experiment-log.md) 에 있다.

## 팩터 라이브러리 — 158개

팩터를 158개 함수로 구현하지 않는다. **선언적 표현식(DSL)** 으로 쓰면 TTM·QoQ·YoY·가속 파생형이 자동 생성된다.

```python
from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.dsl.registry import factor

# 현금 기반 영업수익성 (Ball et al. 2016)
CBOP = factor(
    "CBOP",
    (F.gp - _delta(F.receivables) - _delta(F.inventory) + _delta(F.liabilitiesc)) / F.assets,
    category="quality",
    direction=1,
    neutralize=("sector",),   # 섹터 중립화
)
```

| 카테고리 | 개수 | 예 |
|---|---|---|
| quality | 55 | GP_A, ROIC, F-Score, 발생액, 순영업자산 |
| growth | 26 | 매출·이익 YoY/QoQ |
| price | 24 | 모멘텀 1/3/6/12개월, 12-1, 저변동성 |
| value_price | 24 | PER, PBR, PSR, PFCR, PGPR |
| acceleration | 15 | 성장률의 2차 미분 |
| value_ev | 9 | EV/EBITDA, EV/GP |
| flow_proxy | 5 | 13F 기관 보유 변화, 내부자 순매수 |

문헌 근거가 있는 것만 담는다 — Novy-Marx(2013), Sloan(1996), Hirshleifer et al.(2004), Daniel & Titman(2006), Ball et al.(2016), Chen & Zimmermann 오픈소스 라이브러리 복제 등.

## 사용법

> **데이터 요건.** 팩터 엔진은 [Sharadar](https://sharadar.com) 구독이 있어야 돌아간다 (Bundle, 월 $29~). 개인 가격대에서 **PIT 재무 + 상장폐지 종목**을 함께 주는 곳이 여기뿐이다. 없으면 엔진은 돌지만 돌릴 데이터가 없다. 벤더는 중립 `Provider` 프로토콜 뒤에 격리돼 있어 교체 시 파일 하나만 다시 쓰면 된다. VAA 쪽은 무료 yfinance 라 구독이 필요 없다.

```bash
# 데이터 적재 (Sharadar 구독 필요)
export SHARADAR_API_KEY=...
opt-factor ingest --store us.duckdb --provider sharadar \
  --tables sf1,sep,daily,actions,sp500,tickers --tickers-file universe.txt

# 팩터 예측력 검증 — 10분할 · IC · 회전율
uv run python scripts/factor_lab.py --store us.duckdb --factors GP_A,PER,SIZE

# 공식 성과 (walk-forward + DSR)
opt-factor optimize --store us.duckdb \
  --config configs/strategy_quantus_timed.json \
  --space configs/space_small.json --objective calmar

# 오늘 살 종목 (현재 보유를 주면 매매 계획까지)
opt-factor holdings --store us.duckdb \
  --config configs/strategy_quantus_timed.json --current 내보유.csv

# 운용 화면
opt-factor-tui --store us.duckdb --config configs/strategy_quantus_timed.json
```

전략은 JSON 한 파일로 완전히 선언된다.

```jsonc
{
  "factors": ["PER", "PSR", "POR", "PGPR",
              "NETINC_GROWTH_YOY", "OPINC_GROWTH_YOY",
              "GP_GROWTH_YOY", "REVENUE_GROWTH_YOY"],
  "universe": {
    "min_mcap_usd": 5000000, "max_mcap_usd": 80000000,
    "exclude_financials": true, "exclude_distressed": true
  },
  "backtest": {
    "n_stocks": 20, "rebalance": "QE", "weighting": "equal",
    "hold_multiple": 1.0,                      // 매매 유예구간
    "max_sector_weight": 1.0,                  // 섹터 비중 상한 (1 미만이면 물린다)
    "cost": {"commission_bps": 50, "slippage_bps": 0}
  },
  "timing_ma_days": 200,                       // 마켓타이밍 오버레이 (켜짐/꺼짐)
  "target_vol": null,                          // 연속 변동성 타게팅
  "select_top_k": 0                            // >0 이면 학습 구간 내 팩터 선택
}
```

### 포트폴리오 구성 기법 — 무엇을 만들었고 무엇이 살아남았나

**구현과 채택은 다르다.** 아래는 전부 테스트와 함께 들어와 있고, 판정 칸은
walk-forward 가 이 유니버스에서 내린 답이다.

| 기법 | 판정 |
|---|---|
| 마켓타이밍 오버레이 (Faber 200일 이평) | **채택** — 낙폭 −63.8% → −23.7% |
| 균등가중 | **채택** — 최적화 6종을 전부 이겼다 (DeMiguel 1/N) |
| 매매 유예구간 (`hold_multiple`) | **기각** — 회전율 −23% 대신 수익 −0.86%p |
| 레짐 조건부 팩터 가중 | **기각** — 16.90% → 15.45%, 레짐당 표본 부족 |
| 변동성 타게팅 (Moreira & Muir 2017) | **기각** — 단독으로 쓰면 타이밍을 아예 안 한 것보다 나쁘다 (Sharpe 0.513 → 0.396) |
| 파라미터 앙상블 (`--ensemble k`) | **기각** — 표에서 CAGR 은 제일 높지만 낙폭 −23.7 → −30.6%, Calmar 0.71 → 0.60 |
| 섹터 비중 상한 (`max_sector_weight`) | **성과 중립** — 차이가 0 과 구분되지 않는다 (t = 0.77). 수익 장치가 아니라 위험 장치로 남긴다 |

뒤 세 개는 정교해 보여서가 아니라 **측정이 가리켜서** 만들었다 — 섹터 상한은
실제 보유 포트폴리오가 Technology 32% 로 드러난 뒤에 썼다. 그건 아무도 선택하지
않은 매크로 베팅이다.

## 검증 도구

| 도구 | 무엇을 답하는가 |
|---|---|
| `scripts/factor_lab.py` | 이 팩터에 예측력이 있는가 (10분할 스프레드 · 단조성 · 회전율) |
| `research/ic.py` | Rank IC · IC-IR · 감쇠 프로파일 |
| `research/overfitting.py` | **이 성과가 우연인가** — DSR · PBO(CSCV) |
| `research/regime.py` | 어느 국면에서 작동하는가 (추세 × 변동성 2×2) |
| `research/selection.py` | 팩터 선택을 학습 구간 안에서 — 조합 탐색의 정직한 형태 |
| `optimize/walkforward.py` | 확장·롤링 윈도, 엠바고, 폴드별 파라미터 안정성 |

비중 스킴은 7종이다 — 균등 · 시총 · 역변동성 · 리스크패리티 · HRP · MVO · Black-Litterman.
다만 **이 저장소의 실증 결과는 균등가중이 이긴다** (DeMiguel et al. 2009 의 1/N 결과가 두 번 재현됐다).

---

# 2. VAA 자산배분

Wouter Keller 의 Vigilant Asset Allocation 을 구현하고 walk-forward 로 검증한다.

### 모멘텀 점수 (Keller 13612)

```
momentum = 12·R(1M) + 4·R(3M) + 2·R(6M) + 1·R(12M)
```

### 선택 규칙

- **공격 유니버스**(`SPY`, `EFA`, `EEM`, `AGG`) 중 모멘텀 1위를 선택한다.
- 단, 공격 자산 중 **하나라도 절대 모멘텀이 음수**면 위험 회피 신호로 보고 **방어 유니버스**(`LQD`, `IEF`, `SHY`) 중 1위로 전환한다.
- VAA 선택분 50%, 코어(`SPY`·`TLT`·`GLD`·`BIL`) 각 12.5% 배분(조정 가능).

### 결과

![15년 VAA 전략 비교](backtest_comparison.png)

2011–2026년 15년, 초기 $10,000 기준. 표준 VAA(`Current`)가 ~$29k 로 OU 예측 변형(~$24–27k)을 **앞선다** — 예측 레이어를 더한다고 나아지지 않았다.

```bash
make run                     # 인터랙티브 메뉴
python3 run.py --backtest    # 동적 VAA 백테스트
python3 run.py --optimize    # Sharpe 비중 최적화
```

---

## 구조

```
src/opt_portfolio/
├── factor/                    # 미국 주식 팩터 엔진
│   ├── data/                  #   벤더 어댑터 · PIT 스토어 (DuckDB)
│   ├── dsl/                   #   표현식 트리 · PIT 평가 컨텍스트 · 레지스트리
│   ├── library/               #   팩터 158개 선언
│   ├── universe/              #   유동성·시총·섹터 필터
│   ├── portfolio/             #   스코어 합성 · 비중 7종 · 수축 공분산
│   ├── backtest/              #   횡단면 백테스트 · 비용 · 마켓타이밍
│   ├── optimize/              #   walk-forward · grid/random/GP-EI 탐색
│   ├── research/              #   IC · 분위수 · DSR/PBO · 레짐 · 팩터 선택
│   ├── holdings.py            #   오늘 살 종목 · 매매 계획
│   └── tui.py                 #   운용 화면
├── strategies/                # VAA — 모멘텀 · 자산 선택 · OU 예측(실험)
├── analysis/                  # 백테스트 · 최적화 · 리스크 · 성과
├── core/                      # DuckDB 증분 캐시 · 포지션
└── config.py                  # frozen dataclass 설정
```

## 설치 & 개발

```bash
make install        # uv sync --extra dev
make test           # pytest + 커버리지 (297 tests)
make lint           # ruff check + format --check
make typecheck      # mypy src/
```

패키지 관리는 **uv** 다 (`uv.lock`). `pip install` 하지 않는다.

## 문서

| 파일 | 내용 |
|---|---|
| [`00-overview.md`](docs/factor-system/00-overview.md) | 설계 개요 · 데이터 소스 선정 근거 |
| [`01-factor-spec.md`](docs/factor-system/01-factor-spec.md) | 팩터 정의 |
| [`02-universe-spec.md`](docs/factor-system/02-universe-spec.md) | 유니버스 필터 |
| [`04-data-contract.md`](docs/factor-system/04-data-contract.md) | **스토어 스키마 · PIT 규약 · 벤더 실측 · 운영 절차** |
| [`05-math-spec.md`](docs/factor-system/05-math-spec.md) | 가중·백테스트·walk-forward 수식 |
| [`06-provider-review.md`](docs/factor-system/06-provider-review.md) | 데이터 벤더 12종 비교 |
| [`07-experiment-log.md`](docs/factor-system/07-experiment-log.md) | **실험 기록 — 채택 전략 · 기각 목록 · 재현 절차** |

## 한계와 가정

**팩터 엔진**

- **초소형주 유니버스**의 호가 스프레드는 백테스트 가정(수수료 0.5%·슬리피지 0)보다 비쌀 수 있다. 이 가정은 데이터가 아니다.
- **용량 제약** — 가치가중으로 바꾸면 Sharpe 가 28% 깎인다. 알파가 작은 종목에 있다는 뜻이고, 자금이 커지면 이 성과는 실현되지 않는다.
- **팩터 선택은 DSR 의 시도 횟수에 포함되지 않는다.** 124개를 훑어 고른 행위 자체는 정산되지 않았다 (`research/selection.py` 가 이를 갚기 위한 장치다).
- 세금 미반영.

**VAA**

- 최적화 비중은 in-sample 값이다. 강건성을 함께 봐야 한다.
- 거래비용 0.1% 고정, yfinance 일간 종가 기준, 무위험 수익률 5% 가정, 단일 15년 윈도우.

> ⚠️ 모든 백테스트는 과거 데이터 기준이며 미래 수익을 보장하지 않는다.

## 라이선스

MIT

---

## ⭐ 도움이 되셨다면

이 프로젝트가 유용했다면 우측 상단 **[⭐ Star](https://github.com/younghwan91/opt_portfolio)** 를 눌러주세요. 검색·추천 노출이 올라가 더 많은 분들이 찾을 수 있습니다.

- 🐛 버그·질문 → [Issues](https://github.com/younghwan91/opt_portfolio/issues)
- 📈 업데이트 소식 → [팔로우 @younghwan91](https://github.com/younghwan91)

## 관련 프로젝트 — 한국 주식 퀀트 스택

시세·펀더멘탈·뉴스 수집 REST API부터 데이터 파이프라인, 백테스트·알파 리서치까지 이어지는 오픈소스 스택의 일부입니다.

| 프로젝트 | 설명 |
|---|---|
| **[kiwoom-rest-api](https://github.com/younghwan91/kiwoom-rest-api)** | 키움증권 REST API Python 라이브러리 — 207개 엔드포인트 + 실시간 WebSocket |
| **[krx-fundamentals-api](https://github.com/younghwan91/krx-fundamentals-api)** | 국내 기업 펀더멘탈 REST API — 재무제표·투자지표·배당·종목 스크리닝 (DART + KRX + 네이버) |
| **[krx-news-rest-api](https://github.com/younghwan91/krx-news-rest-api)** | 한국 주식 뉴스·공시 수집 REST API (FastAPI + Redis) |
| **[kr-quant-airflow](https://github.com/younghwan91/kr-quant-airflow)** | 시세·수급·실적 데이터를 TimescaleDB로 수집하는 Airflow 파이프라인 |
| **[kr-quant](https://github.com/younghwan91/kr-quant)** | 코스피·코스닥 알파 리서치 — walk-forward·랜덤 음성대조를 강제하는 검증 가드레일 |
| **[quantbox-engine](https://github.com/younghwan91/quantbox-engine)** | 암호화폐 선물 백테스트·실행 엔진 — 룩어헤드 0, 백테스트↔실거래 일체화 |
| **[automated-stock-trading-systems](https://github.com/younghwan91/automated-stock-trading-systems)** | Bensdorp의 7개 비상관 트레이딩 시스템 백테스터 (교육용 재구현) |

## 만든 사람

**채영환 (Younghwan Chae)** · [GitHub @younghwan91](https://github.com/younghwan91) · [LinkedIn](https://www.linkedin.com/in/younghwan-chae/)

전체 오픈소스 퀀트 스택은 [프로필](https://github.com/younghwan91)에서 한눈에 볼 수 있습니다.
