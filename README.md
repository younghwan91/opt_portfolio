# 🚀 최적 포트폴리오 관리 시스템 v2.0

**동적 VAA (Vigilant Asset Allocation) 전략 + Sharpe Ratio 기반 비중 최적화**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 목차

- [새로운 기능 (v2.0)](#-새로운-기능-v20)
- [핵심 개선사항](#-핵심-개선사항)
- [프로젝트 구조](#-프로젝트-구조)
- [설치 방법](#-설치-방법)
- [빠른 시작](#-빠른-시작)
- [동적 VAA 백테스팅](#-동적-vaa-백테스팅)
- [비중 최적화](#-비중-최적화)
- [API 레퍼런스](#-api-레퍼런스)
- [테스트](#-테스트)
- [변경 이력](#-변경-이력)

---

## 🎉 새로운 기능 (v2.0)

### 1. 동적 VAA 선택 (Dynamic VAA Selection)

**기존 문제점:**
- 고정된 50% VAA 비중으로만 백테스트 가능
- 매월 어떤 ETF가 선택되는지 추적 불가

**새로운 해결책:**
```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

engine = BacktestEngine()
result = engine.run_dynamic_vaa_backtest(years=15)

# VAA 선택 이력 확인
print(result.get_selection_summary())
# 출력 예시:
# SPY    35%
# AGG    25%
# IEF    20%
# EFA    15%
# SHY     5%
```

**특징:**
- ✅ 매월 모멘텀 기반으로 공격형/방어형 ETF 자동 선택
- ✅ 선택 이력 추적 및 분석
- ✅ 방어 모드 비율 계산
- ✅ 월간 리밸런싱 시뮬레이션

### 2. Sharpe Ratio 기반 비중 최적화

**기존 문제점:**
- 50% / 12.5% / 12.5% / 12.5% / 12.5% 고정 비중
- 최적 배분 비율을 찾을 방법 없음

**새로운 해결책:**
```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

engine = BacktestEngine()
result, optimization = engine.run_optimized_backtest(years=15)

# 최적 비중 출력
print(optimization.get_summary())
# 출력 예시:
# 🎯 Optimal Allocation:
#    VAA Selected: 45.0%
#    SPY: 15.0%
#    TLT: 20.0%
#    GLD: 10.0%
#    BIL: 10.0%
#
# 📈 Performance Metrics:
#    Sharpe Ratio: 1.523
#    Annual Return: 12.45%
#    Annual Volatility: 8.17%
#    Max Drawdown: 15.32%
```

**특징:**
- ✅ 그리드 서치로 전역 최적해 탐색
- ✅ Sharpe Ratio 최대화 목표
- ✅ 제약 조건: 비중 합 = 100%, 최소 5% ~ 최대 70%
- ✅ 상위 5개 조합 분석 제공

### 3. 유연한 비중 설정

**기존 문제점:**
- `AllocationConfig`가 `frozen=True`로 수정 불가
- 커스텀 비중 테스트 불가능

**새로운 해결책:**
```python
from src.opt_portfolio.config import AllocationConfig

# 방법 1: 팩토리 메서드
custom_config = AllocationConfig.from_weights(
    vaa=0.40, spy=0.15, tlt=0.20, gld=0.15, bil=0.10
)

# 방법 2: 직접 생성
config = AllocationConfig(
    VAA_SELECTED_WEIGHT=0.45,
    SPY_WEIGHT=0.15,
    TLT_WEIGHT=0.20,
    GLD_WEIGHT=0.10,
    BIL_WEIGHT=0.10
)

# 유효성 검증
assert config.validate()  # 합이 1.0인지 확인
```

**특징:**
- ✅ 각 자산별 개별 비중 설정 가능
- ✅ 자동 검증 (합이 100%인지 확인)
- ✅ 백테스트 엔진에 직접 주입 가능

---

## 🔧 핵심 개선사항

### 코드 정리 (Cleanup)

**삭제된 중복 파일:**
- ❌ `vaa_agg.py` → `src/opt_portfolio/strategies/vaa.py`로 통합
- ❌ `port_ratio_calculator.py` → `src/opt_portfolio/core/portfolio.py`로 통합
- ❌ `rebalance.py` → 기능이 백테스트 엔진에 포함됨
- ❌ `backtest_comparison.py` → 새로운 백테스트 엔진으로 대체
- ❌ `portfolio_ui.py` → `src/opt_portfolio/ui/streamlit_app.py` 사용
- ❌ `integrated_portfolio.py` → CLI로 통합
- ❌ `main.py` → `run.py`로 통합

**결과:**
- 📦 7개 레거시 파일 제거
- 🎯 명확한 단일 진입점 (`run.py`)
- 📚 체계적인 패키지 구조

### 새로운 모듈

#### `src/opt_portfolio/analysis/optimizer.py`

포트폴리오 비중 최적화 엔진

**주요 클래스:**
- `PortfolioOptimizer`: 그리드 서치 기반 최적화
- `OptimizationResult`: 최적화 결과 컨테이너

**주요 메서드:**
```python
class PortfolioOptimizer:
    def generate_weight_combinations(self) -> List[Dict[str, float]]
    def calculate_portfolio_returns(self, vaa_returns, core_returns, weights)
    def calculate_sharpe_ratio(self, returns) -> Tuple[float, float, float]
    def optimize(self, vaa_returns, core_returns) -> OptimizationResult
```

---

## 📁 프로젝트 구조

```
opt_portfolio/
├── run.py                      # 🚀 메인 진입점 (리팩토링됨)
├── test_manual.py              # 🧪 수동 테스트 스크립트
│
├── src/opt_portfolio/          # 메인 패키지
│   ├── config.py              # ⚙️ 설정 (개선: 동적 비중 지원)
│   │
│   ├── core/                  # 핵심 모듈
│   │   ├── cache.py           # DuckDB 캐싱
│   │   └── portfolio.py       # 포트폴리오 관리
│   │
│   ├── strategies/            # 거래 전략
│   │   ├── vaa.py            # VAA 전략
│   │   ├── momentum.py       # 모멘텀 계산
│   │   └── ou_process.py     # OU 프로세스 예측
│   │
│   ├── analysis/              # 분석 모듈
│   │   ├── backtest.py       # 🆕 백테스트 엔진 (대폭 개선)
│   │   ├── optimizer.py      # 🆕 비중 최적화 엔진
│   │   ├── risk.py           # 리스크 지표
│   │   └── performance.py    # 성과 분석
│   │
│   ├── ui/                    # 사용자 인터페이스
│   │   ├── streamlit_app.py  # 웹 UI
│   │   └── cli.py            # CLI
│   │
│   └── utils/                 # 유틸리티
│       ├── helpers.py
│       └── visualization.py
│
└── tests/                      # 테스트 스위트
    └── test_config.py         # 🆕 설정 테스트
```

---

## 🛠️ 설치 방법

### 사전 요구사항
- Python 3.10 이상
- 의존성 패키지 (requirements.txt)

### 설치 단계

```bash
# 1. 저장소 클론
git clone https://github.com/younghwan91/opt_portfolio.git
cd opt_portfolio

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 설치 확인
python run.py
```

---

## 🚀 빠른 시작

### 1. 인터랙티브 메뉴

```bash
python run.py
```

메뉴가 표시됩니다:
```
🚀 OPTIMAL PORTFOLIO MANAGEMENT SYSTEM
   VAA Strategy with Dynamic Selection & Weight Optimization
============================================================

Choose an option:
1. 🌐 Launch Web UI
2. 💻 Launch CLI
3. 📊 Quick VAA Analysis
4. 📈 Run Dynamic VAA Backtest
5. 🔬 Run Optimized Backtest (Sharpe Ratio)
6. 📉 Compare VAA Strategies
7. 📊 Plot Momentum History
8. 💾 Cache Management
9. ❌ Exit
```

### 2. 명령줄 옵션

```bash
# 웹 UI 실행
python run.py --web

# 동적 VAA 백테스트 (15년)
python run.py --backtest

# Sharpe Ratio 최적화
python run.py --optimize
```

---

## 📈 동적 VAA 백테스팅

### 기본 사용법

```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

# 1. 엔진 생성
engine = BacktestEngine()

# 2. 동적 VAA 백테스트 실행 (기본 비중)
result = engine.run_dynamic_vaa_backtest(years=15)

# 3. 결과 확인
print(f"CAGR: {result.cagr:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")

# 4. VAA 선택 이력
selection_summary = result.get_selection_summary()
print(selection_summary)
```

### 커스텀 비중으로 백테스트

```python
# 커스텀 비중 정의
custom_weights = {
    'VAA': 0.40,    # VAA 선택 자산 40%
    'SPY': 0.15,    # S&P 500  15%
    'TLT': 0.20,    # 장기 국채 20%
    'GLD': 0.15,    # 금 15%
    'BIL': 0.10     # 단기 국채 10%
}

# 백테스트 실행
result = engine.run_dynamic_vaa_backtest(
    years=15,
    allocation_weights=custom_weights
)
```

### 결과 분석

```python
# 기본 메트릭
print(f"Initial Capital: ${result.initial_capital:,.0f}")
print(f"Final Capital: ${result.final_capital:,.0f}")
print(f"Total Return: {result.total_return:.2%}")
print(f"CAGR: {result.cagr:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Volatility: {result.volatility:.2%}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Calmar Ratio: {result.calmar_ratio:.3f}")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Defensive Ratio: {result.defensive_ratio:.2%}")

# VAA 선택 분포
selection_counts = pd.Series(result.vaa_selections).value_counts()
print("\nVAA Selection Distribution:")
for ticker, count in selection_counts.items():
    pct = count / len(result.vaa_selections) * 100
    print(f"  {ticker}: {count} months ({pct:.1f}%)")

# 차트 그리기
engine.plot_results({'Dynamic VAA': result})
```

---

## 🔬 비중 최적화

### Sharpe Ratio 최적화

```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

# 1. 엔진 생성
engine = BacktestEngine()

# 2. 최적화 백테스트 실행
result, optimization = engine.run_optimized_backtest(years=15)

# 3. 최적 비중 확인
print(optimization.get_summary())

# 4. 상위 5개 조합 확인
top5 = optimization.all_results.head(5)
print("\nTop 5 Weight Combinations:")
print(top5[['VAA', 'SPY', 'TLT', 'GLD', 'BIL', 'sharpe_ratio']])

# 5. 최적 설정 저장
optimal_config = optimization.optimal_config
```

### 수동 최적화

```python
from src.opt_portfolio.analysis.optimizer import PortfolioOptimizer

# 1. 컴포넌트 수익률 계산 (VAA 선택 + 핵심 자산)
vaa_returns, core_returns = engine._get_component_returns(years=15)

# 2. 옵티마이저 생성
optimizer = PortfolioOptimizer(
    weight_min=0.05,   # 최소 5%
    weight_max=0.70,   # 최대 70%
    weight_step=0.05,  # 5% 단위
    risk_free_rate=0.05
)

# 3. 최적화 실행
opt_result = optimizer.optimize(vaa_returns, core_returns)

# 4. 결과 확인
print(f"Best Sharpe: {opt_result.best_sharpe:.3f}")
print(f"Best Weights: {opt_result.best_weights}")
```

### 최적화 제약 조건

**기본 설정:**
- VAA 비중: 20% ~ 70%
- 핵심 자산 비중: 5% ~ 35%
- 합계: 정확히 100%
- 그리드 단위: 5%

**커스터마이징:**
```python
optimizer = PortfolioOptimizer(
    weight_min=0.10,   # 최소 10%
    weight_max=0.60,   # 최대 60%
    weight_step=0.10,  # 10% 단위 (더 빠름)
    risk_free_rate=0.04
)
```

---

## 📚 API 레퍼런스

### AllocationConfig

```python
from src.opt_portfolio.config import AllocationConfig

# 기본 생성
config = AllocationConfig()

# 커스텀 비중
config = AllocationConfig.from_weights(
    vaa=0.45, spy=0.15, tlt=0.15, gld=0.15, bil=0.10
)

# 속성
config.VAA_SELECTED_WEIGHT  # float
config.SPY_WEIGHT           # float
config.TLT_WEIGHT           # float
config.GLD_WEIGHT           # float
config.BIL_WEIGHT           # float

# 메서드
config.validate()           # bool: 비중 합이 1.0인지 확인
config.target_allocations   # Dict[str, float]: 모든 비중
config.core_weights         # Dict[str, float]: 핵심 자산만
```

### BacktestEngine

```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

# 생성
engine = BacktestEngine(
    initial_capital=10000.0,
    transaction_cost=0.001,
    allocation_config=None  # Optional custom config
)

# 동적 VAA 백테스트
result = engine.run_dynamic_vaa_backtest(
    years=15,
    allocation_weights=None  # Optional custom weights
)

# 최적화 백테스트
result, opt_result = engine.run_optimized_backtest(years=15)

# 전략 비교 백테스트
results = engine.run_vaa_backtest(
    years=15,
    strategies=['Current', 'Forecast_1M', 'Forecast_3M', 'Delta']
)

# 차트 그리기
engine.plot_results(results)
```

### PortfolioOptimizer

```python
from src.opt_portfolio.analysis.optimizer import PortfolioOptimizer

# 생성
optimizer = PortfolioOptimizer(
    weight_min=0.05,
    weight_max=0.70,
    weight_step=0.05,
    risk_free_rate=0.05
)

# 최적화 실행
opt_result = optimizer.optimize(vaa_returns, core_returns)

# 결과 속성
opt_result.best_weights      # Dict[str, float]
opt_result.best_sharpe       # float
opt_result.best_return       # float
opt_result.best_volatility   # float
opt_result.best_max_drawdown # float
opt_result.all_results       # DataFrame: 모든 조합
opt_result.optimal_config    # AllocationConfig
```

---

## 🧪 테스트

### 수동 테스트 실행

```bash
python test_manual.py
```

출력 예시:
```
🧪 PORTFOLIO OPTIMIZATION TEST SUITE
============================================================

TEST 1: Configuration Module
============================================================
✓ Default config created
✓ Config validation passed
✓ Custom config created (VAA 40%, others 15%)
✓ Asset Universe...
✅ Configuration tests PASSED

TEST 2: Portfolio Optimizer
============================================================
✓ Optimizer created
✓ Generated 1234 weight combinations
✓ Portfolio returns calculated
✓ Sharpe ratio calculated
✅ Optimizer tests PASSED

...

============================================================
TEST SUMMARY
============================================================
✅ Passed: 4
❌ Failed: 0
Total: 4
============================================================

🎉 ALL TESTS PASSED! 🎉
```

### pytest 실행 (의존성 있을 경우)

```bash
pytest tests/ -v
```

---

## 📊 사용 예시

### 예시 1: 기본 동적 백테스트

```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

engine = BacktestEngine()
result = engine.run_dynamic_vaa_backtest(years=10)

print(f"10년 백테스트 결과:")
print(f"  최종 자본: ${result.final_capital:,.0f}")
print(f"  CAGR: {result.cagr:.2%}")
print(f"  Sharpe: {result.sharpe_ratio:.3f}")
```

### 예시 2: 최적 비중 찾기

```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

engine = BacktestEngine()
result, optimization = engine.run_optimized_backtest(years=15)

# 최적 비중을 config 파일에 반영
optimal_config = optimization.optimal_config
print(f"최적 VAA 비중: {optimal_config.VAA_SELECTED_WEIGHT:.1%}")
```

### 예시 3: 커스텀 비중 비교

```python
from src.opt_portfolio.analysis.backtest import BacktestEngine

engine = BacktestEngine()

# 시나리오 1: 보수적 (VAA 30%)
conservative = {
    'VAA': 0.30, 'SPY': 0.175, 'TLT': 0.175, 'GLD': 0.175, 'BIL': 0.175
}
result1 = engine.run_dynamic_vaa_backtest(years=10, allocation_weights=conservative)

# 시나리오 2: 공격적 (VAA 60%)
aggressive = {
    'VAA': 0.60, 'SPY': 0.10, 'TLT': 0.10, 'GLD': 0.10, 'BIL': 0.10
}
result2 = engine.run_dynamic_vaa_backtest(years=10, allocation_weights=aggressive)

print(f"보수적: Sharpe {result1.sharpe_ratio:.3f}, MDD {result1.max_drawdown:.2%}")
print(f"공격적: Sharpe {result2.sharpe_ratio:.3f}, MDD {result2.max_drawdown:.2%}")
```

---

## 🔄 변경 이력

### v2.0.0 (2025-01-31)

#### 추가
- ✨ **동적 VAA 백테스팅**: 매월 모멘텀 기반 자산 선택
- ✨ **Sharpe Ratio 최적화**: 그리드 서치 기반 비중 최적화
- ✨ **유연한 비중 설정**: `AllocationConfig.from_weights()` 메서드
- ✨ **VAA 선택 추적**: `BacktestResult.vaa_selections` 및 `get_selection_summary()`
- ✨ **최적화 모듈**: `src/opt_portfolio/analysis/optimizer.py`
- ✨ **수동 테스트**: `test_manual.py` 스크립트
- ✨ **단위 테스트**: `tests/test_config.py`

#### 변경
- 🔧 `AllocationConfig`: `frozen=False`로 변경, 동적 비중 지원
- 🔧 `BacktestEngine`: 동적 VAA 및 최적화 메서드 추가
- 🔧 `run.py`: 메뉴 옵션 확장 (백테스트, 최적화)

#### 제거
- ❌ `vaa_agg.py` (중복)
- ❌ `port_ratio_calculator.py` (중복)
- ❌ `rebalance.py` (통합됨)
- ❌ `backtest_comparison.py` (대체됨)
- ❌ `portfolio_ui.py` (중복)
- ❌ `integrated_portfolio.py` (통합됨)
- ❌ `main.py` (run.py로 통합)

### v1.0.0 (2025-01-15)

- 초기 VAA 전략 구현
- OU 프로세스 예측
- 기본 백테스팅
- Streamlit UI

---

## ⚠️ 주의사항

### 과적합 (Overfitting) 경고

**최적화 결과는 과거 데이터 기반입니다.**
- In-sample 성과 ≠ Out-of-sample 성과
- 최적화된 비중이 미래에도 최적이라는 보장 없음
- Walk-forward 분석 또는 Out-of-sample 테스트 권장

**권장 사항:**
```python
# In-sample 최적화 (2010-2020)
result1, opt1 = engine.run_optimized_backtest(years=10)

# Out-of-sample 테스트 (2020-2025)
# 최적화된 비중을 새로운 기간에 적용
result2 = engine.run_dynamic_vaa_backtest(
    years=5, 
    allocation_weights=opt1.best_weights
)

# 성과 비교
print(f"In-sample Sharpe: {result1.sharpe_ratio:.3f}")
print(f"Out-of-sample Sharpe: {result2.sharpe_ratio:.3f}")
```

### 거래 비용

**백테스트는 월간 0.1% 거래비용을 가정합니다.**
- 실제 브로커 수수료는 다를 수 있음
- 슬리피지는 고려하지 않음
- 세금은 고려하지 않음

### 데이터 품질

**Yahoo Finance 데이터 한계:**
- 생존 편향 (Survivorship Bias)
- 배당금 재투자 가정
- 조정 종가 사용

---

## 📜 라이선스

이 프로젝트는 **MIT 라이선스** 하에 제공됩니다.

---

## 🙏 감사의 말

- **Wouter Keller**: VAA 전략 프레임워크
- **Yahoo Finance**: 시장 데이터
- **오픈소스 커뮤니티**: 훌륭한 도구들

---

## 📞 문의

- GitHub Issues: [https://github.com/younghwan91/opt_portfolio/issues](https://github.com/younghwan91/opt_portfolio/issues)
- Email: your-email@example.com

---

**⚠️ 면책 조항**

이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다.
- 과거 성과가 미래를 보장하지 않습니다
- 투자는 손실의 위험이 있습니다
- 항상 자격 있는 재정 고문과 상담하세요
- 저자는 재정 손실에 대해 책임지지 않습니다

---

*❤️로 정량화(퀀트) 투자자를 위해 제작됨*

**🎯 포트폴리오를 최적화할 준비가 되셨나요?**  
`python run.py`를 실행하고 시작하세요!
