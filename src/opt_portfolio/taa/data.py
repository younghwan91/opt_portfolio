"""Sharadar 펀드 벌크 → 가격 패널.

**`closeadj`(배당조정) 를 쓴다.** 실측: TLT 2010-01-04 이 close 89.81 /
closeadj 55.163 이다. close 를 쓰면 16년간 −8.6%, closeadj 면 +48.7% 로
채권 ETF 수익이 통째로 뒤집힌다. 이 전략은 시간의 절반을 채권에 머무르므로
치명적이다.

yfinance 가 아니라 로컬 벌크를 쓰는 이유: 구독을 종료해도 남고, 팩터 엔진과
원본이 같아 두 서브시스템의 성과가 비교 가능해진다.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_ZIP = Path.home() / "data/sharadar/raw/funds.csv.zip"

#: 벌크에서 읽는 가격 컬럼. 바꾸지 말 것 — 모듈 docstring 참조.
PRICE_COLUMN = "closeadj"


def load_prices(tickers: list[str], zip_path: Path | None = None) -> pd.DataFrame:
    """요청한 티커의 일별 배당조정 종가 패널.

    Args:
        tickers: 티커 목록
        zip_path: 펀드 벌크 zip (기본 `~/data/sharadar/raw/funds.csv.zip`)

    Returns:
        인덱스 = 거래일(오름차순), 컬럼 = 티커, 값 = `closeadj`

    Raises:
        ValueError: 요청한 티커 중 하나라도 벌크에 없으면. 조용히 빈 컬럼을
            만들면 이후 모든 계산이 NaN 으로 흘러가 결과가 "성공" 으로 보인다.
    """
    path = zip_path or DEFAULT_ZIP
    if not path.exists():
        raise FileNotFoundError(f"펀드 벌크가 없다: {path}")

    wanted = set(tickers)
    rows: list[tuple[str, str, float]] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"zip 안의 csv 가 하나가 아니다: {names}")
        with zf.open(names[0]) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
            for row in reader:
                t = row["ticker"]
                if t not in wanted:
                    continue
                raw = row[PRICE_COLUMN]
                if raw == "":
                    continue
                rows.append((t, row["date"], float(raw)))

    if not rows:
        raise ValueError(f"벌크에서 아무 행도 못 찾았다: {sorted(wanted)}")

    frame = pd.DataFrame(rows, columns=["ticker", "date", "px"])
    panel = frame.pivot(index="date", columns="ticker", values="px")
    panel.index = pd.DatetimeIndex(panel.index)
    panel = panel.sort_index()

    missing = wanted - set(panel.columns)
    if missing:
        raise ValueError(f"벌크에 없는 티커: {sorted(missing)}")

    return panel[sorted(wanted)]
