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

run select "$CONFIGS/strategy_select.json"
run quantus_voltarget "$CONFIGS/strategy_quantus_voltarget.json"
run volonly "$CONFIGS/strategy_volonly.json"
run quantus_sectorcap "$CONFIGS/strategy_quantus_sectorcap.json"
run quantus_ens3 "$CONFIGS/strategy_quantus_timed.json" --ensemble 3
echo "전체 완료 $(date +%H:%M:%S)"
