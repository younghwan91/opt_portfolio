# results/ — 실행 산출물

문서에 적힌 성과 숫자를 **직접 검산할 수 있게** 원자료를 둔다. 표를 믿어달라고
하는 대신 시계열을 주는 편이 낫다.

## 무엇이 있나

| 경로 | 내용 |
|---|---|
| `oos/oos_*.json` | walk-forward **검증 구간 일별 수익률** 35건 — 비용 차감 후 |
| `universes/universe_*.txt` | 각 실험이 쓴 후보 티커 목록 |
| `analysis/` | 팩터 IC·상관·10분할·레짐 IC·보유 종목 |

`oos_*.json` 은 `{unix초: 일별수익률}` 이다. 이름은 `configs/strategy_*.json` 과
대응한다 (`oos_cost_guards.json` ↔ `strategy_cost_guards.json`).

## 검산

```python
import json, pandas as pd, numpy as np
d = json.load(open("results/oos/oos_quantus_train5.json"))
r = pd.Series({pd.Timestamp(int(k), unit="s"): v for k, v in d.items()}).sort_index()

eq   = (1 + r).cumprod()
yrs  = (r.index[-1] - r.index[0]).days / 365.25
cagr = eq.iloc[-1] ** (1 / yrs) - 1
mdd  = (eq / eq.cummax() - 1).min()
print(f"CAGR {cagr:.2%}  MDD {mdd:.1%}  Calmar {cagr/abs(mdd):.2f}")
# CAGR 23.74%  MDD -23.7%  Calmar 1.00
```

문서의 Sharpe 는 무위험수익률(`config.RISK_FREE_RATE`)을 뺀 값이라 위 식과 다르다.
CAGR·MDD·Calmar 는 무위험수익률과 무관하므로 그대로 맞아야 한다.

## 꼭 같이 볼 것

**산출물마다 어떤 방어 장치를 끄고 잰 것인지가 다르다.**

| | |
|---|---|
| `oos_quantus_*` | 채택 계열 — 슬리피지 0 · 최소 주가 0 · 최소 거래대금 0. **실현되지 않는 값이다** |
| `oos_cost_guards` | 같은 전략에 방어를 켠 것 — Sharpe **−0.224**, MDD **−99.2%** |
| `oos_lean_timed_*` | 대형주 E안 — 방어를 켠 채로 DSR 0.996 |

무엇이 왜 그런지는 [`docs/factor-system/07-experiment-log.md`](../docs/factor-system/07-experiment-log.md)
§5.5 · §5.8 에 있다. **숫자만 떼어 쓰지 마시라.**

## 없는 것 — 벤더 원본

Sharadar 벌크(6.7GB)와 그 적재 결과인 DuckDB 스토어(2.1GB)는 **유료 구독물이라
재배포할 수 없다.** 공개 방침의 문제가 아니라 라이선스 문제다.

즉 여기 있는 산출물은 **검산은 되지만 재실행은 안 된다.** 재실행하려면 Sharadar
구독이 필요하고, 적재 절차는 `docs/factor-system/04-data-contract.md` §5 에 있다.

## 데이터 시점

**2026-08-14 에서 멈춰 있다.** 구독을 종료했으므로 이후 갱신은 없다.
`holdings_20260816.csv` 는 그 시점의 보유 목록이며, 이후 시장 상황을 반영하지
않는다 — 매매 근거로 쓰지 마시라.
