#!/bin/sh
# 풀 히스토리 walk-forward 를 **한 프로세스에서 순차로** 돌린다.
#
# 왜 이렇게 쓰는가: 이전에는 `pgrep 이 비면 시작` 형태의 대기자를 여러 개
# 걸어두었는데, 대기자 둘이 같은 조건을 동시에 만족해 **같은 작업을 두 번**
# 띄웠다. 15GB 머신에서 8.5GB 짜리 프로세스가 둘이 되어 OOM 으로 죽었다
# (2026-08-15, 커널 로그 01:01·02:05 두 건).
#
# 대기 조건을 검사하지 않는 것이 요점이다. `;` 로 이은 한 줄은 정의상
# 순차 실행이고, 경쟁할 상대가 없다.
#
#   nohup sh scripts/run_backtests.sh > ~/data/queue.log 2>&1 &
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
STORE=${STORE:-$HOME/data/us_micro.duckdb}
UNIVERSE=${UNIVERSE:-$HOME/data/universe_quantus.txt}
OUT=${OUT:-$HOME/data}
# 전략 설정은 저장소 밖에 있다 — 초소형주 레시피는 체크포인트다
# (`configs/README.md`). 공개 `configs/` 에는 합성 예제 둘만 남는다.
CONFIGS=${CONFIGS:-$HOME/data/configs}

# 시작 전에 벤치마크를 확인한다. 없으면 여기서 죽어야 한다.
#
# 두 번 당했다. 벌크 동기화가 `prices` 를 SEP 로 재구축하면 SFP 에 있는 SPY 가
# 사라지는데, 그 사실을 **몇 시간짜리 큐가 시작하고 나서야** 알았다
# (2026-08-15 band_8_150, 2026-08-16 cost_guards·band_8_150·roll5 3연속).
# 실패는 시끄러웠지만 너무 늦게 시끄러웠다 — 검사를 앞으로 당긴다.
if ! uv run --project "$REPO" python "$REPO/scripts/ensure_benchmark.py" \
        --store "$STORE" --check-only; then
    echo "!!! 벤치마크가 없어 배치를 시작하지 않는다."
    echo "    uv run python scripts/ensure_benchmark.py --store $STORE"
    exit 1
fi

run() {
    name=$1
    config=$2
    shift 2
    if [ -f "$OUT/oos_$name.json" ]; then
        echo "건너뜀 (이미 있음): $name"
        return
    fi
    echo "=== $name 시작 $(date +%H:%M:%S) ==="
    uv run --project "$REPO" opt-factor optimize \
        --store "$STORE" --config "$config" \
        --space "$CONFIGS/space_small.json" \
        --tickers-file "$UNIVERSE" \
        --method grid --trials 3 --min-train-years 5 --objective calmar \
        --out "$OUT/oos_$name.json" "$@" > "$OUT/opt_$name.log" 2>&1
    status=$?
    # 실패를 조용히 넘기지 않는다 — 이 저장소의 지배적 실패 유형이다.
    # OOM 킬은 종료코드 137 로 온다. 이걸 안 보면 다음 작업이 시작되면서
    # "돌고 있다" 는 인상만 남고 결과 파일이 없다는 사실은 몇 시간 뒤에 안다.
    if [ "$status" -ne 0 ]; then
        echo "!!! $name 실패 (종료코드 $status) — $OUT/opt_$name.log 확인"
    else
        grep -E "Deflated|OOS Sharpe" "$OUT/opt_$name.log"
    fi
    echo "=== $name 종료 $(date +%H:%M:%S) ==="
}

# 2026-08-16 배치 — 07-experiment-log §5.5 의 현실적 비용 검증.
#
# 순서에 의도가 있다. 비용 3건을 먼저 돌린다: 슬리피지만 올리는 것과 유동성
# 필터까지 켜는 것을 **따로** 재야 기여를 나눌 수 있고, 셋 중 `cost_guards` 가
# 이 저장소에서 가장 중요한 실험이기 때문이다 — 무너지면 알파가 거래 불가능한
# 종목에만 있었다는 뜻이다.
#
# 뒤의 둘은 어젯밤 미완으로 남은 것이다. band_8_150 은 SPY 가 아직 적재되기
# 전(21:45)에 시작해 벤치마크 없음으로 죽었고, roll5 는 로그가 0바이트다.
run cost_slip50 "$CONFIGS/strategy_cost_slip50.json"
run cost_slip150 "$CONFIGS/strategy_cost_slip150.json"
run cost_guards "$CONFIGS/strategy_cost_guards.json"
run band_8_150 "$CONFIGS/strategy_band_8_150.json"
run roll5 "$CONFIGS/strategy_quantus_timed.json" --train-window 5

# 아래쪽 사다리 — 밴드를 올리면 무너진다는 §3.4 의 반대편.
#
# **예상을 먼저 적는다: 셋 다 백테스트 성과가 오를 것이다.** 그리고 그 개선은
# 이 저장소에서 가장 못 믿을 숫자다. `universe/filters.py` 가 하한을 "선택이
# 아니라 방어" 라고 적어둔 이유가 이것이고(Hou·Xue·Zhang 2020), 체결 불가능한
# 초소형이 극단 분위수를 채우면 균등가중 백테스트는 조용히 부풀려진다.
#
# 그래서 `band_down06_guards` 가 이 묶음의 존재 이유다 — 같은 밴드를 방어
# 켠 채로 돌린다. 켜고도 남으면 진짜고, 켜면 사라지면 위 셋은 신기루다.
# 방어 없는 셋만 돌리고 "더 좋다" 고 적는 것이 정확히 §2.2 가 저지른 실수다.
run band_down06 "$CONFIGS/strategy_band_down06.json"
run band_down04 "$CONFIGS/strategy_band_down04.json"
run band_cap_half "$CONFIGS/strategy_band_cap_half.json"
run band_down06_guards "$CONFIGS/strategy_band_down06_guards.json"

# v2 후보 — 방어를 켜고도 살아남는 밴드가 있는가 (§5.7).
#
# 채택 밴드는 방어를 켜면 후보가 목표 보유 수를 밑돌 만큼 줄어든다.
# 위로 올릴수록 거래 가능한 종목은 늘지만 알파는 사라진다 — ×3 부터 DSR 이
# 관문 아래다. **겹치는 구간은 ×1.6~×2 뿐**이고, 그 둘을 방어 켠 채로 잰다.
#
# 두 조건의 DSR 0.999 는 **방어를 끈 상태의 값**이다. 켜면 유니버스도 수익도
# 달라지므로 추론하지 않고 실제로 돌린다.
run v2_band16_guards "$CONFIGS/strategy_v2_band16_guards.json"
run v2_band20_guards "$CONFIGS/strategy_v2_band20_guards.json"
echo "전체 완료 $(date +%H:%M:%S)"
