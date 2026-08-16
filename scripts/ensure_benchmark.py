#!/usr/bin/env python3
"""
벤치마크 가격이 스토어에 있는지 보장한다 — 벌크 동기화 **이후** 실행한다.

## 왜 별도 단계인가

가격 원본 동기화는 `quant-airflow` 가 맡고, 그쪽은 SEP(주식) 벌크로 `prices`
를 재구축한다. 그런데 **벤치마크 SPY 는 SEP 가 아니라 SFP(펀드) 에 있다.**
따라서 동기화가 정상 동작할수록 SPY 는 매번 사라진다. 원본을 받아오는 쪽의
버그가 아니라, **이 저장소가 필요로 하는 후처리를 이 저장소가 안 하고 있던
것**이다.

실제로 두 번 당했다:

- 2026-08-15 21:45 `band_8_150` — `ValueError: 벤치마크 'SPY' 가격이 없어…`
- 2026-08-16 17:22 동기화 직후 `cost_guards`·`band_8_150`·`roll5` 3건 연속 실패

두 번 다 **몇 시간짜리 큐가 시작하고 나서야** 알았다. 실패는 시끄러웠지만
(그건 잘 설계된 것이다) 너무 늦게 시끄러웠다.

## 사용

    python3 scripts/ensure_benchmark.py --store ~/data/us_micro.duckdb

`--check-only` 는 적재하지 않고 존재 여부만 검사한다. 긴 배치 앞에 두어
**시작 전에** 죽게 하는 용도다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import duckdb

#: 벌크 CSV 컬럼 → 스토어 `prices` 컬럼.
#: `close` 가 **수정주가**(`closeadj`)라는 것이 이 매핑의 요점이다. 실측으로
#: 확인했다 — AAPL 2015-06-15 이 스토어 close 28.288 / closeunadj 126.92 이고
#: 벌크의 closeadj·closeunadj 와 일치한다. 뒤집으면 수익률이 분할만큼 튄다.
COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "closeadj": "close",
    "closeunadj": "closeunadj",
    "volume": "volume",
}

#: 최소 기대 행수. 1997년부터면 7,000 거래일을 넘는다. 이보다 적으면 벌크가
#: 잘렸거나 다른 파일을 보고 있는 것이므로 조용히 넘기지 않는다.
MIN_ROWS = 7_000


def read_from_bulk(zip_path: Path, ticker: str) -> list[tuple[object, ...]]:
    """펀드 벌크(zip)에서 해당 티커 행만 뽑는다."""
    rows: list[tuple[object, ...]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise SystemExit(f"zip 안의 csv 가 하나가 아니다: {names}")
        with zf.open(names[0]) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
            for row in reader:
                if row.get("ticker") != ticker:
                    continue
                rows.append(
                    (
                        ticker,
                        row["date"],
                        *[_num(row.get(src)) for src in COLUMNS],
                    )
                )
    return rows


def _num(value: str | None) -> float | None:
    """Sharadar 직판은 숫자를 문자열로 준다 — 빈 값은 NULL 로."""
    if value is None or value == "":
        return None
    return float(value)


def current_state(con: duckdb.DuckDBPyConnection, ticker: str) -> tuple[int, object, object]:
    row = con.execute(
        "select count(*), min(date), max(date) from prices where ticker = ?", [ticker]
    ).fetchone()
    assert row is not None
    return int(row[0]), row[1], row[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--benchmark", default="SPY", help="보장할 벤치마크 티커")
    ap.add_argument(
        "--funds-zip",
        type=Path,
        default=Path.home() / "data/sharadar/raw/funds.csv.zip",
        help="SFP(펀드) 벌크 zip — 벤치마크는 여기 있다",
    )
    ap.add_argument("--check-only", action="store_true", help="적재하지 않고 검사만")
    args = ap.parse_args()

    if not args.store.exists():
        print(f"!!! 스토어가 없다: {args.store}", file=sys.stderr)
        return 2

    con = duckdb.connect(str(args.store), read_only=args.check_only)
    n, lo, hi = current_state(con, args.benchmark)

    if n >= MIN_ROWS:
        print(f"벤치마크 {args.benchmark}: {n:,}행 {lo} ~ {hi} — 정상")
        return 0

    if args.check_only:
        print(
            f"!!! 벤치마크 {args.benchmark} 가 {n:,}행뿐이다 (기대 {MIN_ROWS:,}+).\n"
            f"    벌크 동기화가 prices 를 SEP 로 재구축하면 SFP 에 있는 벤치마크가 사라진다.\n"
            f"    `python3 scripts/ensure_benchmark.py --store {args.store}` 로 복구한다.",
            file=sys.stderr,
        )
        return 1

    if not args.funds_zip.exists():
        print(f"!!! 펀드 벌크가 없다: {args.funds_zip}", file=sys.stderr)
        return 2

    print(f"벤치마크 {args.benchmark}: {n:,}행 — 복구한다 (원본 {args.funds_zip.name})")
    rows = read_from_bulk(args.funds_zip, args.benchmark)
    if len(rows) < MIN_ROWS:
        print(
            f"!!! 벌크에서 {len(rows):,}행만 나왔다 (기대 {MIN_ROWS:,}+). "
            "적재하지 않는다 — 적은 데이터로 덮으면 더 나쁘다.",
            file=sys.stderr,
        )
        return 1

    cols = ", ".join(["ticker", "date", *COLUMNS.values()])
    placeholders = ", ".join(["?"] * (2 + len(COLUMNS)))
    con.execute("begin")
    con.execute("delete from prices where ticker = ?", [args.benchmark])
    con.executemany(f"insert into prices ({cols}) values ({placeholders})", rows)
    con.execute("commit")

    # 넣었다고 말하기 전에 세어본다.
    n2, lo2, hi2 = current_state(con, args.benchmark)
    if n2 < MIN_ROWS:
        print(f"!!! 적재 후에도 {n2:,}행뿐이다", file=sys.stderr)
        return 1
    print(f"복구 완료: {n2:,}행 {lo2} ~ {hi2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
