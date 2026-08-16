from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from opt_portfolio.taa.data import load_prices

HEADER = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated"


def _make_zip(tmp_path: Path, rows: list[str]) -> Path:
    csv = tmp_path / "funds.csv"
    csv.write_text("\n".join([HEADER, *rows]) + "\n")
    zp = tmp_path / "funds.csv.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(csv, arcname="funds.csv")
    return zp


class TestLoadPrices:
    def test_uses_closeadj_not_close(self, tmp_path: Path) -> None:
        """배당조정가를 써야 한다 — close 를 쓰면 채권 ETF 수익이 뒤집힌다."""
        zp = _make_zip(
            tmp_path,
            [
                "TLT,2010-01-04,0,0,0,89.81,100,55.163,89.81,2026-08-14",
                "TLT,2026-08-14,0,0,0,82.04,100,82.04,82.04,2026-08-14",
            ],
        )
        px = load_prices(["TLT"], zip_path=zp)

        assert px.loc[pd.Timestamp("2010-01-04"), "TLT"] == pytest.approx(55.163)
        assert px.loc[pd.Timestamp("2026-08-14"), "TLT"] == pytest.approx(82.04)

    def test_filters_to_requested_tickers(self, tmp_path: Path) -> None:
        zp = _make_zip(
            tmp_path,
            [
                "TLT,2010-01-04,0,0,0,1,100,55.163,1,2026-08-14",
                "SPY,2010-01-04,0,0,0,1,100,90.0,1,2026-08-14",
                "QQQ,2010-01-04,0,0,0,1,100,40.0,1,2026-08-14",
            ],
        )
        px = load_prices(["TLT", "SPY"], zip_path=zp)

        assert sorted(px.columns) == ["SPY", "TLT"]

    def test_missing_ticker_fails_loudly(self, tmp_path: Path) -> None:
        """조용히 빈 컬럼을 만들지 않는다 — 이 저장소의 지배적 실패 유형이다."""
        zp = _make_zip(tmp_path, ["SPY,2010-01-04,0,0,0,1,100,90.0,1,2026-08-14"])

        with pytest.raises(ValueError, match="NOPE"):
            load_prices(["SPY", "NOPE"], zip_path=zp)

    def test_index_is_sorted_datetime(self, tmp_path: Path) -> None:
        zp = _make_zip(
            tmp_path,
            [
                "SPY,2010-01-05,0,0,0,1,100,91.0,1,2026-08-14",
                "SPY,2010-01-04,0,0,0,1,100,90.0,1,2026-08-14",
            ],
        )
        px = load_prices(["SPY"], zip_path=zp)

        assert isinstance(px.index, pd.DatetimeIndex)
        assert px.index.is_monotonic_increasing
