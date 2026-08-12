"""
수정주가 소급 재조정 탐지 — 증분 적재로 가격이 조용히 썩는 것을 막는다.

벤더의 `closeadj` 는 분할·배당이 생기면 **과거 전체가 다시 계산된다.**
풀 히스토리를 한 번 받아두고 이후 최근 구간만 증분 적재하면:

    [1998 ~ 2021] 스토어에 남은 옛 계수  |  [2021 ~ 현재] 새로 받은 새 계수

경계에서 분할 배수만큼의 **가짜 수익률**이 하루 생긴다. 에러는 나지 않는다.
겹치는 날짜의 비율을 보면 재조정 여부와 그 계수를 알 수 있으므로,
비율이 1 이 아니면 재조정 이전 구간을 같은 계수로 맞춘다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from opt_portfolio.factor.data.store import PITStore


def _prices(ticker: str, dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates),
            "close": closes,
            "closeunadj": closes,
        }
    )


@pytest.fixture
def store():
    with PITStore(":memory:") as s:
        yield s


class TestDetectAdjustmentFactors:
    def test_no_split_gives_no_factor(self, store: PITStore) -> None:
        store.upsert_prices(_prices("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 101.0]))
        incoming = _prices("AAPL", ["2024-01-03", "2024-01-04"], [101.0, 102.0])

        assert store.detect_adjustment_factors(incoming) == {}

    def test_two_for_one_split_detected(self, store: PITStore) -> None:
        """분할 후 벤더는 과거 수정주가를 절반으로 다시 준다."""
        store.upsert_prices(_prices("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 200.0]))
        incoming = _prices("AAPL", ["2024-01-03", "2024-01-04"], [100.0, 110.0])

        factors = store.detect_adjustment_factors(incoming)

        assert factors["AAPL"] == pytest.approx(0.5)

    def test_only_affected_tickers_reported(self, store: PITStore) -> None:
        store.upsert_prices(_prices("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 200.0]))
        store.upsert_prices(_prices("MSFT", ["2024-01-02", "2024-01-03"], [50.0, 60.0]))
        incoming = pd.concat(
            [
                _prices("AAPL", ["2024-01-03"], [100.0]),
                _prices("MSFT", ["2024-01-03"], [60.0]),
            ]
        )

        assert set(store.detect_adjustment_factors(incoming)) == {"AAPL"}

    def test_no_overlap_gives_no_factor(self, store: PITStore) -> None:
        """겹치는 날짜가 없으면 비교할 근거가 없다 — 추측하지 않는다."""
        store.upsert_prices(_prices("AAPL", ["2024-01-02"], [100.0]))
        incoming = _prices("AAPL", ["2024-02-01"], [50.0])

        assert store.detect_adjustment_factors(incoming) == {}


class TestRescalePrices:
    def test_rescale_applies_to_stored_history(self, store: PITStore) -> None:
        store.upsert_prices(_prices("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 200.0]))

        store.rescale_prices({"AAPL": 0.5})

        got = store.conn.execute(
            "SELECT close FROM prices WHERE ticker='AAPL' ORDER BY date"
        ).fetchall()
        assert [r[0] for r in got] == [50.0, 100.0]

    def test_rescale_leaves_unadjusted_close_alone(self, store: PITStore) -> None:
        """closeunadj 는 원시값이라 재조정 대상이 아니다 (페니스톡 필터용)."""
        store.upsert_prices(_prices("AAPL", ["2024-01-02"], [100.0]))

        store.rescale_prices({"AAPL": 0.5})

        got = store.conn.execute("SELECT closeunadj FROM prices WHERE ticker='AAPL'").fetchone()
        assert got[0] == 100.0

    def test_rescale_only_named_tickers(self, store: PITStore) -> None:
        store.upsert_prices(_prices("AAPL", ["2024-01-02"], [100.0]))
        store.upsert_prices(_prices("MSFT", ["2024-01-02"], [100.0]))

        store.rescale_prices({"AAPL": 0.5})

        got = store.conn.execute("SELECT close FROM prices WHERE ticker='MSFT'").fetchone()
        assert got[0] == 100.0


class TestEndToEnd:
    def test_incremental_update_after_split_has_no_fake_return(self, store: PITStore) -> None:
        """
        이 테스트가 이 기능의 존재 이유다.

        100 → 102 (실제 +2%) 로 움직이던 종목이 2:1 분할하면, 벤더는 과거를
        50 → 51 로 다시 준다. 재조정 없이 최근 구간만 덮어쓰면 스토어는
        100 → 51 이 되어 경계에 -50% 라는 가짜 수익률이 생긴다.
        """
        store.upsert_prices(_prices("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 102.0]))
        incoming = _prices("AAPL", ["2024-01-03", "2024-01-04"], [51.0, 51.5])

        store.rescale_prices(store.detect_adjustment_factors(incoming))
        store.upsert_prices(incoming)

        closes = pd.Series(
            [
                r[0]
                for r in store.conn.execute(
                    "SELECT close FROM prices WHERE ticker='AAPL' ORDER BY date"
                ).fetchall()
            ]
        )
        returns = closes.pct_change().dropna()

        assert (returns.abs() < 0.30).all(), f"경계에 가짜 수익률이 남았다: {list(returns)}"


class TestNullKeyRows:
    """
    벤더 데이터에 키가 빈 행이 실재한다 (DAILY 벌크 CSV 300만 행당 약 44건).

    NOT NULL 제약에 걸려 적재 전체가 죽거나, 조용히 사라지거나 둘 중
    하나인데 둘 다 안 된다 — 걸러내되 반드시 로그를 남긴다.
    """

    def test_null_ticker_rows_are_dropped_with_warning(self, store: PITStore, caplog) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", None],
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "close": [100.0, 50.0],
            }
        )

        with caplog.at_level("WARNING"):
            n = store.upsert_prices(df)

        assert n == 1, "정상 행은 들어가야 한다"
        assert "키 결측" in caplog.text, "조용히 버리면 안 된다"
