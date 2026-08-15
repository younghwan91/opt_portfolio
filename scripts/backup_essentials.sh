#!/bin/sh
# 기계를 잃어도 복구 불가능한 것만 묶는다.
#
#   sh scripts/backup_essentials.sh /media/외장디스크
#
# 왜 이것만 묶는가: `~/data` 는 6.4GB 지만 그중 6.2GB 는 스토어와 벤더 벌크이고
# **구독만 유지되면 다시 받을 수 있다.** 재취득이 불가능한 것은 실험 좌표와
# 산출물 약 3MB 뿐이다. 큰 것을 같이 묶으면 백업이 번거로워져서 결국 안 돌린다.
#
# 암호화하지 않는다. 목적지가 자신의 외장 디스크라는 전제다. 클라우드에 둘
# 거라면 이 파일을 `age -p` 로 한 번 더 감싼다 — **공개 저장소에는 넣지 않는다.**
# 암호문은 지울 수 없고, 오늘 안전한 암호가 5년 뒤에도 안전하리라는 보장이 없다.
set -u

DEST=${1:-}
SRC=${DATA_DIR:-$HOME/data}
[ -z "$DEST" ] && { echo "사용법: sh $0 <백업 폴더>"; exit 2; }
[ -d "$DEST" ] || { echo "백업 폴더가 없다: $DEST"; exit 2; }
[ -d "$SRC" ] || { echo "데이터 폴더가 없다: $SRC"; exit 2; }

STAMP=$(date +%Y-%m-%d)
OUT="$DEST/opt_portfolio_$STAMP.tgz"

# 묶을 대상. 스토어(`*.duckdb`)와 벌크(`sharadar/`)는 의도적으로 뺀다.
ITEMS="results logs configs analysis README.md"
GLOBS="universe_*.txt"

missing=""
for i in $ITEMS; do
    [ -e "$SRC/$i" ] || missing="$missing $i"
done
# 조용한 절단 금지 — 없는 것을 빼고 "성공" 으로 끝내지 않는다.
[ -n "$missing" ] && echo "!!! 없는 항목:$missing (계속 진행하되 백업은 불완전하다)"

set -- $ITEMS
for g in $GLOBS; do
    for f in "$SRC"/$g; do
        [ -e "$f" ] && set -- "$@" "$(basename "$f")"
    done
done

tar czf "$OUT" -C "$SRC" "$@" || { echo "!!! tar 실패"; exit 1; }

# 만들었다고 말하기 전에 실제로 들어갔는지 센다.
n=$(tar tzf "$OUT" | wc -l)
size=$(du -h "$OUT" | cut -f1)
echo "백업: $OUT  ($size · 항목 $n개)"
[ "$n" -lt 10 ] && { echo "!!! 항목이 너무 적다 — 경로를 확인하라"; exit 1; }

echo
echo "이 백업에 **없는** 것:"
echo "  - 스토어(us_micro.duckdb) · 벤더 벌크(sharadar/) → 재다운로드 가능"
echo "  - SHARADAR_API_KEY → 비밀번호 관리자에 따로 보관"
exit 0
