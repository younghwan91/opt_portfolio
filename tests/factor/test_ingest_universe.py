"""
유니버스 지정 경로 — 풀 히스토리 적재의 입구.

`accessible_tickers()` 자동 탐색은 최근 분기 재무가 있는 종목만 찾으므로
상장폐지 종목이 원리적으로 빠진다. 유료 플랜의 폐지 종목까지 적재하려면
유니버스를 **밖에서 명시**해야 하고, 18,000종목을 명령줄로 넘길 수는 없다.
따라서 이 경로가 막히면 생존편향 제거가 통째로 무산된다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opt_portfolio.factor.cli import _read_ticker_file


class TestTickerFile:
    def test_reads_bulk_csv_ticker_column(self, tmp_path: Path) -> None:
        path = tmp_path / "tickers.csv"
        path.write_text("ticker,name,isdelisted\nAAPL,Apple,N\nENRN,Enron,Y\n")

        assert _read_ticker_file(path) == ["AAPL", "ENRN"]

    def test_reads_plain_text_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "universe.txt"
        path.write_text("aapl\nmsft\n nvda \n")

        assert _read_ticker_file(path) == ["AAPL", "MSFT", "NVDA"]

    def test_deduplicates(self, tmp_path: Path) -> None:
        path = tmp_path / "universe.txt"
        path.write_text("AAPL, AAPL, MSFT")

        assert _read_ticker_file(path) == ["AAPL", "MSFT"]

    def test_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="유니버스 파일이 없습니다"):
            _read_ticker_file(tmp_path / "nope.txt")

    def test_empty_file_fails_loudly(self, tmp_path: Path) -> None:
        """빈 유니버스로 진행하면 0종목 적재가 '성공'으로 끝난다."""
        path = tmp_path / "empty.txt"
        path.write_text("   \n")

        with pytest.raises(SystemExit, match="비었습니다"):
            _read_ticker_file(path)

    def test_csv_without_ticker_column_fails_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.csv"
        path.write_text("symbol,name\nAAPL,Apple\n")

        with pytest.raises(SystemExit, match="ticker"):
            _read_ticker_file(path)
