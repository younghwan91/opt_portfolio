# 수학 프로토콜 스펙 — 검증 · 최적화 · 정산

이 문서는 시스템의 수학적 결정 사항과 그 근거를 기록한다.
**사용자의 역할은 `run_walk_forward()` 를 호출하는 것뿐이다** — 나머지 규율은 코드가 강제한다.

---

## 1. 체결 규약 (전 모듈 공통)

```
신호: 리밸런싱일 t 종가까지의 정보          (scores.loc[t])
체결: t+1 종가                             (engine._select → exec_date)
수익: t+1 → t+2 부터 새 비중으로 귀속       (engine 세그먼트)
```

`research/ic.py` 의 `forward_returns` 와 `backtest/engine.py` 가 **같은 규약**을 쓴다.
IC 로 검증한 예측력이 백테스트에서 그대로 재현되게 하기 위함이다 — 규약이 어긋나면
IC 는 좋은데 백테스트가 안 되는(또는 그 반대) 원인 불명의 괴리가 생긴다.

## 2. 팩터 검증 (research/)

| 도구 | 수식 | 판정 기준 |
|---|---|---|
| Rank IC | 날짜별 Spearman(스코어, 순방향수익) | 겹침보정 t ≥ 2 |
| IC-IR | mean(IC)/std(IC) | ≥ 0.3 양호 |
| 분위 스프레드 | Q10 − Q1 동일가중 수익 | t ≥ 3 + 단조성 ≥ 0.8 |
| 회전율 | 상위 분위 월간 교체율 | 스프레드 ÷ (회전율×비용) > 2 |

일별 IC 는 h일 보유 시 h−1 일이 겹치므로 유효표본을 n/h 로 줄여 t 를 계산한다 (`summarize_ic`).

## 3. 스코어 합성 (portfolio/score.py)

- 정규화: **rank-normal** (rank → Φ⁻¹). z-score 대비 극단값 강건.
- 합성: 2단계 — 카테고리 내 균등 평균 → 카테고리 간 가중.
  (가치 31개 팩터가 31표를 갖는 왜곡 방지)
- IC 가중은 **트레일링만** (`trailing_ic_weights(as_of=...)`), 음수 IC 는 0 클립.
- `min_coverage`: 관측 팩터 가중치 합이 50% 미만인 종목은 랭킹에서 제외.

## 4. 비중 결정 (portfolio/weights.py)

| 스킴 | 정식화 | 추정 대상 |
|---|---|---|
| equal | 1/N | 없음 (기준선) |
| inverse_vol | wᵢ ∝ 1/σᵢ | 분산만 |
| risk_parity | min ½wᵀΣw − Σbᵢln wᵢ (Spinu, 볼록) | Σ |
| hrp | 클러스터링 + 재귀 이분할 (역행렬 불사용) | Σ (강건) |
| mvo | max μᵀw − (δ/2)wᵀΣw, μ = z·ir_scale | μ, Σ |
| black_litterman | μ_post = π + τΣ(τΣ+Ω)⁻¹(Q−π), Q = π + ir_scale·z | μ, Σ, 확신도 |

- 공분산은 전 스킴 공통 **Ledoit-Wolf 상수상관 수축** (`covariance.py`).
  수축 강도 δ* 는 닫힌형 — **하이퍼파라미터가 아니므로 PO 대상에서 제외**한다.
- BL 의 `view_confidence` 가 "팩터를 얼마나 믿는가"의 연속 다이얼:
  0 → 시장 포트폴리오, ∞ → 순수 팩터 베팅. **PO 의 핵심 축.**
- 모든 스킴 출력은 `cap_and_normalize` (water-filling) 로 상한 준수.

## 5. 파라미터 최적화 — 유일한 공식 경로

```python
from opt_portfolio.factor.optimize.walkforward import run_walk_forward

result = run_walk_forward(
    evaluate,            # (params, start, end) → 일별 수익률 (백테스트 클로저)
    space={              # 탐색 공간 — 이것이 곧 전략의 자유도
        "n_stocks":        ("int",   10, 50),
        "weighting":       ("cat",   ["equal", "hrp", "black_litterman"]),
        "view_confidence": ("float", 0.05, 2.0),
        "rebalance":       ("cat",   ["ME", "QE"]),
    },
    calendar=trading_days,
    method="bayesian",         # GP-EI: 적은 시도 = 낮은 SR₀ = 유리한 DSR
    n_trials_per_fold=24,
    embargo_days=21,           # ≥ 보유기간
)

result.oos_returns        # 공식 성과 (train 에 쓰인 적 없는 수익률만)
result.sharpe()           # OOS Sharpe
result.deflated_sharpe()  # 전 시도 횟수로 정산한 유의확률 — 0.95 이상만 신뢰
result.param_stability()  # 폴드별 선택 파라미터 — 널뛰면 노이즈 피팅 경고
```

### 규율이 코드에 박힌 지점

1. **전체 표본 최적화 API 가 없다.** `search()` 는 있지만 OOS 성과를 만들지 않으므로
   보고할 수 없다. 공식 성과는 `WalkForwardResult.oos_returns` 뿐.
2. **모든 시도가 기록된다.** `SearchResult.trials` — 최고값만 돌려주는 인터페이스는
   의도적으로 만들지 않았다. n_trials 없이는 DSR 을 계산할 수 없기 때문.
3. **embargo** 가 train/test 사이 순방향 수익률 겹침을 차단한다 (purged CV 논리).
4. **실패한 시도는 −∞ 로 기록**되어 탐색은 계속되되 통계에는 남는다.

### 판정 절차 (전략 승인 조건)

| 단계 | 조건 | 실패 시 |
|---|---|---|
| ① OOS Sharpe | > 0.5 | 폐기 |
| ② Deflated Sharpe | > 0.95 | "통계적으로 우연과 구분 불가" — 폐기 |
| ③ 파라미터 안정성 | 폴드 간 IQR 이 공간의 1/3 이내 | 해당 축 고정 후 재실험 |
| ④ PBO (시도 로그 재활용) | < 0.3 | 탐색 공간 축소 후 재실험 |
| ⑤ 비용 민감도 | 비용 2배에도 Sharpe > 0.3 | 실전 투입 불가 (kr-quant 의 교훈) |

## 6. 과최적화 정산 수식 (research/overfitting.py)

**Deflated Sharpe** (Bailey & López de Prado 2014):

```
SR₀ = √V[SR] · [(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))]      γ = 0.5772…
DSR = Φ( (SR−SR₀)√(T−1) / √(1 − γ₃SR + (γ₄−1)/4·SR²) )
```

V[SR] 는 walk-forward 시도 로그에서 실측한다 (`WalkForwardResult.deflated_sharpe`).

**PBO** (CSCV): 표본을 S=10 블록으로 나눠 C(10,5)=252 조합의 IS/OOS 반전마다
IS 최적 전략의 OOS 상대순위 ω 를 구하고, ω < 0.5 빈도를 PBO 로 한다.
블록 단위 분할인 이유: 일별 셔플은 자기상관을 파괴해 PBO 를 과소평가한다.

## 7. 의도된 비대칭 — 무엇을 PO 하고 무엇을 하지 않는가

| PO 대상 (walk-forward 안) | PO 금지 (구조로 고정) |
|---|---|
| n_stocks, rebalance 주기 | 체결 규약 (t+1) |
| 비중 스킴, view_confidence, ir_scale | LW 수축 강도 (닫힌형) |
| 카테고리 가중치, 팩터 선택 | 비용 모델 계수 (실측값) |
| 타이밍 on/off, reentry_days | embargo (보유기간이 결정) |
| max_weight, cov_window | winsorize 상수 (검증 리포트가 결정) |

비용·체결·수축까지 PO 에 넣으면 최적화기는 반드시 "비용이 낮고 미래를 보는" 설정을
찾아낸다. 자유도는 전략 쪽에만 주고, 물리 법칙 쪽은 잠근다.
