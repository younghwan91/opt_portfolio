# 🚀 최적 포트폴리오 관리 시스템

**Vigilant Asset Allocation (VAA)** 전략과 **Ornstein-Uhlenbeck (OU) 프로세스 예측**을 기반으로 한 전문가급 정량화(퀀트) 포트폴리오 관리 시스템입니다. 자동 리밸런싱, 위험 분석, 백테스팅 기능을 제공합니다.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 목차

- [주요 기능](#-주요-기능)
- [프로젝트 구조](#-프로젝트-구조)
- [설치 방법](#-설치-방법)
- [빠른 시작](#-빠른-시작)
- [전략 개요](#-전략-개요)
- [퀀트 전문가 인사이트](#-퀀트-전문가-인사이트)
- [API 레퍼런스](#-api-레퍼런스)
- [성과](#-성과)
- [기여](#-기여)

---

## ✨ 주요 기능

### 핵심 기능

| 기능 | 설명 |
|------|------|
| 🔍 **VAA 선택** | 다중 기간 모멘텀 분석 기반 자동 ETF 선택 |
| 🔮 **OU 예측** | 평균 회귀 모델링 및 몬테카를로 시뮬레이션 |
| ⚡ **스마트 캐싱** | DuckDB 기반 증분식 데이터 수집 |
| ⚖️ **자동 리밸런싱** | 정수 주식 최적화 및 현금흐름 관리 |
| 📊 **리스크 분석** | Sharpe, Sortino, VaR, CVaR, 최대낙폭 등 |
| 📈 **백테스팅** | 다중 전략 비교 및 거래비용 포함 |
| 🌐 **웹 UI** | Streamlit 대시보드 및 Plotly 차트 |
| 💻 **CLI** | 완전한 명령줄 인터페이스 |

### 고급 분석 기능

- **다중 전략 비교**: 현재, 1M/3M/6M 예측, 모멘텀 변화율(Δ)
- **승률 계산**: 최고 수익 자산이 될 확률 (몬테카를로 기반)
- **시장 구간 분석**: 상승/하락 시장 수익 비율
- **낙폭 분석**: 상위 낙폭 기간 및 회복 시간
- **성과 분석**: 연도별 및 시장 구간별 세부 분석

---

## 📁 프로젝트 구조

```
opt_portfolio/
├── src/opt_portfolio/          # 메인 패키지
│   ├── __init__.py            # 패키지 초기화
│   ├── config.py              # 설정 및 상수
│   │
│   ├── core/                  # 핵심 모듈
│   │   ├── cache.py           # DuckDB 캐싱 시스템
│   │   └── portfolio.py       # 포트폴리오 관리
│   │
│   ├── strategies/            # 거래 전략
│   │   ├── vaa.py            # VAA 전략 구현
│   │   ├── momentum.py       # 모멘텀 계산
│   │   └── ou_process.py     # OU 프로세스 예측
│   │
│   ├── analysis/              # 분석 모듈
│   │   ├── backtest.py       # 백테스팅 엔진
│   │   ├── risk.py           # 리스크 지표
│   │   └── performance.py    # 성과 분석
│   │
│   ├── ui/                    # 사용자 인터페이스
│   │   ├── streamlit_app.py  # 웹 UI
│   │   └── cli.py            # 명령줄 인터페이스
│   │
│   └── utils/                 # 유틸리티
│       ├── helpers.py        # 헬퍼 함수
│       └── visualization.py  # 차트 유틸
│
├── tests/                     # 테스트 스위트
├── docs/                      # 문서
├── run.py                     # 메인 진입점
├── pyproject.toml            # 프로젝트 설정
└── README.md                 # 이 파일
```

---

## 🛠️ 설치 방법

### 사전 요구사항
- Python 3.10 이상
- pip 패키지 매니저

### 설치 단계

1. **저장소 클론:**
```bash
git clone https://github.com/younghwan91/opt_portfolio.git
cd opt_portfolio
```

2. **가상환경 생성 (권장):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

3. **의존성 설치:**
```bash
pip install -e .
# 또는 개발 모드
pip install -e ".[dev]"
```

4. **설치 확인:**
```bash
python run.py
```

---

## 🚀 빠른 시작

### 방법 1: 웹 UI (권장)

```bash
python run.py --web
# 또는
streamlit run src/opt_portfolio/ui/streamlit_app.py
```

### 방법 2: 명령줄 인터페이스

```bash
python run.py --cli
```

### 방법 3: Python API

```python
from opt_portfolio.strategies.vaa import VAAStrategy
from opt_portfolio.core.portfolio import Portfolio
from opt_portfolio.analysis.backtest import BacktestEngine

# VAA 분석 실행
vaa = VAAStrategy(use_forecasting=True)
result = vaa.select()
print(f"선택된 ETF: {result.selected_etf}")
print(f"시장 모드: {'방어' if result.is_defensive else '성장'}")

# 승률 계산
win_probs, forecast = vaa.get_win_probabilities(months=1)
print(f"승률:\n{win_probs}")
```

### 레거시 인터페이스 (여전히 사용 가능)

- **VAA 분석만**: `python vaa_agg.py`
- **리밸런싱 계산기**: `python rebalance.py`
- **백테스트 비교**: `python backtest_comparison.py`

---

## 📊 전략 개요

### VAA (Vigilant Asset Allocation)

VAA는 **Wouter Keller**가 2017년에 개발한 전술적 자산배분 전략입니다.

#### 자산군

| 자산군 | ETF | 용도 |
|--------|-----|------|
| **공격형** | SPY, EFA, EEM, AGG | 강세장에서의 성장 |
| **방어형** | LQD, IEF, SHY | 약세장에서의 자본 보호 |
| **핵심 자산** | SPY, TLT, GLD, BIL | 전략적 항상 보유 |

#### 목표 배분

```
┌─────────────────────────────────────────┐
│                                         │
│    ┌─────────────────────┐              │
│    │  VAA 선택 ETF      │    50%        │
│    │   (전술적 배분)    │              │
│    └─────────────────────┘              │
│                                         │
│    ┌─────┬─────┬─────┬─────┐            │
│    │ SPY │ TLT │ GLD │ BIL │  각 12.5%  │
│    │     │     │     │     │            │
│    └─────┴─────┴─────┴─────┘            │
│         (핵심 자산)                      │
│                                         │
└─────────────────────────────────────────┘
```

#### 모멘텀 공식

가중 모멘텀 점수 계산:

$$\text{Momentum Score} = 12 \times r_{1m} + 4 \times r_{3m} + 2 \times r_{6m} + 1 \times r_{12m}$$

여기서 $r_{nm}$ = n개월 수익률(%)

#### 선택 로직

```python
IF any(공격형 모멘텀 < 0):
    모드 = 방어
    선택 = argmax(방어형 모멘텀)
ELSE:
    모드 = 성장
    선택 = argmax(공격형 모멘텀)
```

### 🔮 고급 예측 및 백테스팅

정교한 예측 엔진이 포함되어 있습니다:

| 전략 | 설명 | 15년 수익률 |
|------|------|-----------|
| **표준 VAA** | 현재 점수가 최고인 자산 선택 | **+114.6%** |
| **1개월 예측** | 다음달 점수 예측으로 선택 | **+173.7%** |
| **모멘텀 변화(Δ)** | 모멘텀 증가율이 최고인 자산 선택 | **+201.3%** |
| **3개월 예측** | 3개월 후 점수 예측으로 선택 | **+238.8%** |
| **6개월 예측** | 6개월 후 점수 예측으로 선택 | **+242.2%** |

*주의: 과거 성과가 미래를 보장하지 않습니다.*

---

## 🎓 퀀트 전문가 인사이트

### 1. 모멘텀의 학술적 배경

모멘텀은 학술적으로 가장 강력하게 검증된 시장 이상현상(market anomaly) 중 하나입니다.

> **"승자는 계속 승리하고, 패자는 계속 패배한다"** - Jegadeesh & Titman (1993)

**VAA 가중치 (12, 4, 2, 1) 근거:**
- 모멘텀의 반감기(half-life): 약 3-6개월
- 단기 모멘텀에 높은 가중치 → 빠른 시장 반응
- 장기 모멘텀 포함 → 노이즈 필터링

### 2. OU 프로세스 (Ornstein-Uhlenbeck Process)

모멘텀 점수는 장기적으로 0 주변으로 회귀하는 경향이 있습니다.

$$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$$

| 파라미터 | 의미 | 전형적 범위 |
|---------|------|-----------|
| θ (theta) | 평균 회귀 속도 | 0.001 - 0.1 |
| μ (mu) | 장기 평균 | ~ 0 |
| σ (sigma) | 변동성 | 자산별 |

**캘리브레이션 (Calibration):**
AR(1) 회귀를 통해 파라미터 추정:
- $\beta = e^{-\theta}$
- $\alpha = \mu(1 - \beta)$

### 3. 리밸런싱 최적화

**정수 주식 제약:**
- 완벽한 목표 배분은 불가능
- 우선순위: 큰 편차부터 교정
- 매도 후 매수 순서로 현금 흐름 최적화

**권장 리밸런싱 주기:**

| 주기 | 장점 | 단점 |
|------|------|------|
| 일별 | 최적 추적 | 거래비용 과다 |
| 주별 | 균형 | 노이즈 거래 |
| **월별** | **비용 효율적** | **약간의 추적 오차** |
| 분기별 | 최소 비용 | 큰 편차 가능 |

### 4. 리스크 지표 해석

| 지표 | 좋음 | 보통 | 주의 |
|------|------|------|------|
| Sharpe Ratio | > 2.0 | 1.0 - 2.0 | < 1.0 |
| 최대낙폭 | < 15% | 15-25% | > 25% |
| Calmar Ratio | > 1.5 | 1.0 - 1.5 | < 1.0 |
| 승률 | > 60% | 50-60% | < 50% |

### 5. 백테스트 주의사항

⚠️ **과적합 (Overfitting) 경고:**
- In-sample 성과 ≠ Out-of-sample 성과
- 파라미터 최적화 → 과적합 위험
- Walk-forward 분석 권장

⚠️ **생존 편향 (Survivorship Bias):**
- 상장폐지된 종목 누락 → 성과 과대평가
- ETF는 상대적으로 안전

⚠️ **미래 정보 누설 (Look-Ahead Bias):**
- 미래 데이터 사용 → 비현실적 성과
- 월말 가격만 사용 (조정 종가)

### 6. 실전 적용 가이드

**최소 자본금 권장:**
```
$10,000 이상 (배분 오차 < 3%)
$50,000 이상 (배분 오차 < 1%)
```

**거래 비용:**
- ETF 스프레드: ~0.01%
- 수수료: $0 (대부분 브로커)
- 총 예상 비용: 리밸런싱당 ~0.1%

**세금 고려:**
- 월별 리밸런싱 → 단기 양도소득
- 세금 이연 계좌 활용 권장 (IRA, 401k 등)

---

## 📚 API 레퍼런스

### VAAStrategy

```python
from opt_portfolio.strategies.vaa import VAAStrategy

vaa = VAAStrategy(
    aggressive_tickers=['SPY', 'EFA', 'EEM', 'AGG'],
    protective_tickers=['LQD', 'IEF', 'SHY'],
    use_cache=True,
    use_forecasting=True
)

# 선택 실행
result = vaa.select(calculation_date=date.today())

# 승률 계산
win_probs, forecast_df = vaa.get_win_probabilities(months=1)
```

### Portfolio

```python
from opt_portfolio.core.portfolio import Portfolio

portfolio = Portfolio.from_dict({'SPY': 100, 'TLT': 50})
portfolio.update_prices()

# 현재 배분 조회
allocation = portfolio.get_allocation()

# 리밸런싱 계산
recommendations = portfolio.calculate_rebalance(
    selected_etf='AGG',
    additional_cash=10000
)
```

### BacktestEngine

```python
from opt_portfolio.analysis.backtest import BacktestEngine

engine = BacktestEngine(
    initial_capital=10000,
    transaction_cost=0.001  # 0.1%
)

results = engine.run_vaa_backtest(years=15)
engine.plot_results(results)
```

### RiskAnalyzer

```python
from opt_portfolio.analysis.risk import RiskAnalyzer

analyzer = RiskAnalyzer(risk_free_rate=0.05)
metrics = analyzer.calculate_all_metrics(returns=monthly_returns)
print(analyzer.get_risk_report(metrics))
```

---

## 🛠️ 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| **numpy** | ≥1.24.0 | 수치 계산 |
| **pandas** | ≥2.0.0 | 데이터 조작 및 분석 |
| **yfinance** | ≥0.2.36 | 실시간 금융 데이터 |
| **streamlit** | ≥1.28.0 | 웹 UI 프레임워크 |
| **plotly** | ≥5.18.0 | 인터랙티브 차트 |
| **duckdb** | ≥0.9.0 | 고속 칼럼 캐싱 |
| **scipy** | ≥1.11.0 | 통계 분석 |

---

## 🚨 중요 사항

- **📊 데이터 출처**: Yahoo Finance API로 실시간 가격 수집
- **🕐 시장 시간**: 정확한 가격을 위해 시장 시간 중 사용 권장
- **🔄 리밸런싱 주기**: 월 1회 권장
- **⚠️ 위험 고지**: 이 소프트웨어는 교육용이며 재정 조언이 아닙니다

---

## 🤝 기여

버그 수정, 새 기능, 문서 개선, 추가 테스트에 대한 Pull Request를 환영합니다!

1. 저장소를 Fork합니다
2. Feature 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 오픈합니다

---

## ⚠️ 면책 조항

**이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다.**

- 과거 성과가 미래를 보장하지 않습니다
- 투자는 손실의 위험이 있습니다
- 항상 자격 있는 재정 고문과 상담하세요
- 저자는 재정 손실에 대해 책임지지 않습니다

---

## 📜 라이선스

이 프로젝트는 오픈소스이며 **MIT 라이선스** 하에 제공됩니다. [LICENSE](LICENSE) 파일을 참고하세요.

---

## 🙏 감사의 말

- Wouter Keller (VAA 전략 프레임워크)
- Yahoo Finance (시장 데이터)
- 오픈소스 커뮤니티 (훌륭한 도구들)

---

*❤️로 정량화(퀀트) 투자자를 위해 제작됨*

**🎯 포트폴리오를 최적화할 준비가 되셨나요?** `python run.py`를 실행하고 원하는 인터페이스를 선택하세요!
