#!/usr/bin/env python3
"""
Sharadar 직판 API 실응답을 테스트 픽스처로 녹화한다.

벤더 동작을 추측해서 코드를 쓰지 않는다는 규약(CLAUDE.md)의 집행 도구다.
녹화된 픽스처는 `tests/factor/test_vendor_contract.py` 가 재생하며,
벤더가 스키마를 바꾸면 그 시점에 테스트가 깨진다.

사용법:
    export SHARADAR_API_KEY=...
    uv run python scripts/record_sharadar_fixtures.py

기본은 소량(티커 2개 × 짧은 구간)만 받는다. 구독 쿼터를 거의 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "factor" / "fixtures" / "sharadar"

#: (테이블, 티커, 추가 파라미터) — 문서화된 함정을 전부 포함하도록 고른다.
#: SF1 은 ARQ 평균컬럼 null 확인용, DAILY 는 백만달러 단위 확인용,
#: SF2 는 문자열 숫자 확인용, SF3A 는 date→calendardate 복원 확인용.
TARGETS: list[tuple[str, list[str], dict[str, Any]]] = [
    ("SF1", ["AAPL", "MSFT"], {"dimension": "ARQ", "from": "2024-01-01"}),
    ("SEP", ["AAPL", "MSFT"], {"from": "2025-06-01"}),
    ("DAILY", ["AAPL", "MSFT"], {"from": "2025-06-01"}),
    ("SF2", ["AAPL"], {"from": "2025-01-01"}),
    ("SF3A", ["AAPL"], {"from": "2024-01-01"}),
    ("TICKERS", ["AAPL", "MSFT"], {}),
]

#: 픽스처에서 제거할 키 — API 키가 파일에 새어나가지 않게 한다.
_SECRET_PARAMS = {"api_key"}


def _redact(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in _SECRET_PARAMS}


def record(limit: int, verbose: bool) -> int:
    from opt_portfolio.factor.data.sharadar import (
        _DIRECT_TABLES,
        _DIRECT_URL,
        _default_get_json,
    )

    api_key = os.environ.get("SHARADAR_API_KEY") or os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        print("SHARADAR_API_KEY 가 설정되지 않았습니다.", file=sys.stderr)
        return 1

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for table, tickers, extra in TARGETS:
        slug = _DIRECT_TABLES.get(table, table.lower())
        url = _DIRECT_URL.format(table=slug)
        params: dict[str, Any] = {
            **extra,
            "ticker": ",".join(tickers),
            "api_key": api_key,
            "format": "json",
            "limit": limit,
        }
        try:
            payload = _default_get_json(url, params)
        except Exception as exc:  # 녹화 실패는 개별로 보고하고 계속한다
            print(f"  ✗ {table}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue

        rows = payload if isinstance(payload, list) else payload.get("data", [])
        out = FIXTURE_DIR / f"{table.lower()}.json"
        out.write_text(
            json.dumps(
                {
                    "table": table,
                    "url": url,
                    "params": _redact(params),
                    "recorded_rows": len(rows),
                    "payload": payload,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"  ✓ {table}: {len(rows)}행 → {out.relative_to(Path.cwd())}")
        if verbose and rows:
            first = rows[0] if isinstance(rows[0], dict) else None
            if first:
                print(f"      컬럼: {sorted(first)}")

    if failures:
        print(f"\n{failures}개 테이블 녹화 실패", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200, help="테이블당 최대 행 수 (기본 200)")
    ap.add_argument("-v", "--verbose", action="store_true", help="컬럼 목록도 출력")
    args = ap.parse_args()

    print(f"픽스처 녹화 → {FIXTURE_DIR}")
    return record(args.limit, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
