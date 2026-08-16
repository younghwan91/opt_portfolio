# configs/ — 전략 설정

**전부 공개한다.** 채택 전략, 기각된 후보, 검증 실험까지 이 디렉터리의 JSON
하나면 각각을 그대로 재현할 수 있다.

| 파일 | 무엇인가 |
|---|---|
| `strategy.json` · `space.json` | 스키마 전 필드를 보여주는 기본 예제 (중대형주 밴드) |
| `strategy_quantus*.json` | 채택 계열 — 초소형주 8팩터, `_timed` 가 200일 타이밍 포함 |
| `strategy_lean*.json` | 대형주(역대 S&P500) 5팩터 — `_timed` 가 E안 |
| `strategy_mid*.json` · `strategy_micro.json` · `strategy_small_agg.json` | 유니버스 사다리 실험 |
| `strategy_band_*.json` | 시총 밴드 사다리 (`07-experiment-log.md` §3.4) |
| `strategy_cost_*.json` | 현실적 비용 검증 (§5.5) |
| `strategy_v2_*.json` | 방어 켠 밴드 후보 (§5.7) |
| `strategy_select*.json` | 학습 구간 내 팩터 선별 |

```bash
opt-factor optimize --store us.duckdb \
  --config configs/strategy_quantus_timed.json \
  --space configs/space_small.json --objective calmar
```

## 왜 감추지 않는가

2026-08-15 에는 감췄다. 초소형주 전략은 **용량이 극도로 작아서** 같은 종목을 사는
사람이 늘면 자기 체결가가 밀리고, 종목만 감춰도 **시총 밴드와 팩터 조합이 종목을
결정하기 때문에 레시피가 있으면 같은 종목이 나온다.** 그 논리는 지금도 맞다.

**틀린 것은 그게 지킬 만한 물건이라는 전제였다.**

2026-08-16 검증에서 두 가지가 드러났다:

1. **채택 전략은 방어 장치를 켜면 무너진다** (§5.5) — 최소 주가·최소 거래대금을
   켜면 유니버스의 98% 가 사라져 Sharpe −0.22 · DSR 0.002 다. 시총 밴드와 유동성
   필터가 서로 배타적이다.
2. **용량 상한이 1억 남짓이다** — 실제 보유 종목의 일 거래대금 중앙값이 약 $45k,
   그중 둘은 아예 0 이었다.

즉 감춘 레시피는 **실제로 굴릴 수 있는 물건이 아니었다.** 반대로 실제로 쓸 만한
대형주 전략(`strategy_lean_timed.json`)은 용량 제약이 없어 **남이 따라 사도 체결가가
밀리지 않는다.**

> 감춘 것이 쓸 것을 보호하지 않았고, 쓸 것은 감출 이유가 없었다.

## 숫자를 그대로 믿지는 마시라

설정마다 **어떤 방어를 끄고 잰 것인지**가 다르다. 채택 계열(`strategy_quantus*`)은
슬리피지 0 · 최소 주가 0 · 최소 거래대금 0 이고, 그 상태의 성과는 실현되지 않는다.

어떤 설정이 무엇을 껐고 그래서 결과가 어떻게 달라지는지는
`docs/factor-system/07-experiment-log.md` 에 전부 적혀 있다. 채택 1건 뒤에 기각
20건이 있고, 그 기각의 근거도 같이 있다.
