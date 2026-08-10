"""
Sharadar 직판 API 벤더 계약 검증 — 문서화된 함정이 실제로 처리되는지.

`docs/factor-system/04-data-contract.md` §3 의 **2026-08-05 실계정 실측**을
페이로드로 옮겨 적었다. 이 파일의 페이로드 모양이 곧 "벤더가 이렇게 준다"는
우리의 가정이며, 가정이 틀리면 여기가 아니라 실적재에서 조용히 터진다.

⚠️ 이 픽스처는 **녹화가 아니라 전사(轉寫)**다 — 구독 만료로 재현 호출이
불가능한 상태에서 작성됐다. 구독 후 `scripts/record_sharadar_fixtures.py` 를
실행하면 실응답이 저장되고, `TestRecordedFixtureDrift` 가 전사 내용과
실제 응답이 어긋나는지 검사한다.

여기서 검증하는 함정 (전부 실데이터 버그로 한 번씩 터진 것들):
  1. 숫자가 문자열로 온다 → 합산 전 to_numeric
  2. DAILY 의 marketcap/ev 는 백만 달러 단위 → 1e6 환산
  3. 날짜 컬럼이 전 테이블 `date` 로 통일 → 테이블별 원래 이름 복원
  4. 절단 방향이 ticker 필터 유무로 뒤집힘 → 항상 티커 청크로 요청
  5. SEP 의 close(원시)/closeadj 공존 → closeadj 우선
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from opt_portfolio.factor.data.sharadar import (
    SharadarProvider,
    TruncatedDataError,
    _aggregate_insiders,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sharadar"


def _provider(payloads: list[dict], **kwargs) -> tuple[SharadarProvider, list[dict]]:
    """호출 파라미터를 기록하면서 준비된 페이로드를 순서대로 돌려주는 프로바이더."""
    calls: list[dict] = []
    seq = iter(payloads)

    def fake_get(url: str, params: dict) -> dict:
        calls.append(dict(params))
        return next(seq, {"count": 0, "data": []})

    provider = SharadarProvider(api_key="t", get_json=fake_get, api="direct", **kwargs)
    return provider, calls


class TestVendorPayloadTraps:
    """실측으로 확인된 벤더 함정이 어댑터에서 처리되는지."""

    def test_daily_marketcap_converted_from_millions_to_dollars(self) -> None:
        """
        DAILY 의 marketcap/ev 는 백만 달러 단위다 (SF1 은 달러).

        실측: AAPL marketcap 4,428,166.1 (DAILY) vs 4,508,288,143,800 (SF1).
        미환산 시 PER 등 모든 배수가 10⁶배 틀어진다.
        """
        payload = {
            "count": 1,
            "data": [
                # 숫자가 문자열로 오는 것까지 동시에 재현한다
                {
                    "ticker": "AAPL",
                    "date": "2025-06-02",
                    "marketcap": "4428166.1",
                    "ev": "4500000.0",
                }
            ],
        }
        provider, _ = _provider([payload])
        frame = next(iter(provider.daily_metrics(tickers=["AAPL"])))

        # 4,428,166.1 백만 달러 → 약 4.43조 달러
        assert frame["mcap"].iloc[0] == pytest.approx(4_428_166.1 * 1e6)
        assert frame["mcap"].iloc[0] > 1e12, "환산 누락 시 조 단위에 못 미친다"
        assert frame["ev"].iloc[0] == pytest.approx(4_500_000.0 * 1e6)

    def test_string_numbers_are_coerced_before_summing(self) -> None:
        """
        직판 JSON 은 숫자를 문자열로 준다 — 합산 전 강제 변환이 없으면
        문자열 연결("1000-300")이 되거나 TypeError 로 죽는다.
        """
        raw = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "filingdate": ["2024-01-15", "2024-02-20"],
                "transactionshares": ["1000", "-300"],  # ← 문자열
            }
        )
        out = _aggregate_insiders(raw)
        q1 = out[out["calendardate"] == pd.Timestamp("2024-03-31")]
        assert q1["insider_net_shares"].iloc[0] == 700.0

    @pytest.mark.parametrize(
        ("table_method", "payload_extra", "restored_col"),
        [
            (
                "fundamentals",
                {"calendardate": "2024-03-31", "reportperiod": "2024-03-31"},
                "datekey",
            ),
            ("institutions", {"shrunits": "1000", "shrholders": "5"}, "calendardate"),
        ],
    )
    def test_date_column_restored_per_table(
        self, table_method: str, payload_extra: dict, restored_col: str
    ) -> None:
        """
        직판은 전 테이블의 기준 날짜를 `date` 하나로 통일해 보낸다.
        테이블별 원래 이름(SF1=datekey, SF3A=calendardate)으로 복원돼야
        PIT 검증과 스토어 스키마가 성립한다.
        """
        payload = {"count": 1, "data": [{"ticker": "AAPL", "date": "2024-05-02", **payload_extra}]}
        provider, _ = _provider([payload])
        frame = next(iter(getattr(provider, table_method)(tickers=["AAPL"])))
        assert restored_col in frame.columns, f"{restored_col} 로 복원되지 않았다"

    def test_requests_are_always_ticker_filtered(self) -> None:
        """
        절단 방향이 ticker 필터 유무로 뒤집힌다 (필터 있음=가장 오래된 N행,
        없음=가장 최근 N행). 마칭 로직은 '오래된 쪽부터'를 전제하므로
        모든 요청에 ticker 가 실려야 한다.
        """
        payload = {"count": 1, "data": [{"ticker": "A", "date": "2024-01-02", "closeadj": "1.0"}]}
        provider, calls = _provider([payload] * 3)
        list(provider.prices(tickers=["A", "B", "C"]))

        assert calls, "요청이 한 번도 나가지 않았다"
        for params in calls:
            assert params.get("ticker"), f"ticker 필터 없는 요청: {params}"

    def test_sep_prefers_adjusted_close(self) -> None:
        """SEP 에는 close(원시)와 closeadj 가 공존한다 — 수정주가를 써야 한다."""
        payload = {
            "count": 1,
            "data": [{"ticker": "A", "date": "2024-01-02", "close": "10.0", "closeadj": "5.0"}],
        }
        provider, _ = _provider([payload])
        frame = next(iter(provider.prices(tickers=["A"])))
        assert frame["close"].astype(float).iloc[0] == 5.0, "원시 close 가 남았다"


class TestCursorlessTableTruncation:
    """
    TICKERS 는 날짜 커서가 없어 단일 요청으로 받는다 (`_DIRECT_PAGE_COL`).

    무료 티어(500종목)에서는 이 가정이 성립했지만, 유료 플랜은 폐지 종목을
    포함해 ~18,000종목이라 페이지 한도(10,000)를 넘는다. 커서가 없으니
    이어받을 수단이 없고, 잘린 유니버스로 적재하면 없는 종목이 백테스트에서
    조용히 빠진다 — 구독 첫날 터질 절단이다.
    """

    def test_full_page_without_cursor_raises(self) -> None:
        payload = {"count": 3, "data": [{"ticker": f"T{i}"} for i in range(3)]}
        provider, _ = _provider([payload], page_size=3)

        with pytest.raises(TruncatedDataError, match="TICKERS"):
            provider.tickers()

    def test_partial_page_without_cursor_is_complete(self) -> None:
        """한도 미만이면 전량을 받은 것이다 — 정상 종료."""
        payload = {"count": 2, "data": [{"ticker": "A"}, {"ticker": "B"}]}
        provider, _ = _provider([payload], page_size=3)

        assert len(provider.tickers()) == 2


class TestRecordedFixtureDrift:
    """
    녹화된 실응답이 있으면, 위 전사 내용과 어긋나지 않는지 검사한다.

    구독 후 `uv run python scripts/record_sharadar_fixtures.py` 를 실행하면
    활성화된다. 벤더가 스키마를 바꾸면 여기서 먼저 깨진다.
    """

    @staticmethod
    def _load(table: str) -> list[dict]:
        path = FIXTURE_DIR / f"{table}.json"
        if not path.exists():
            pytest.skip(f"녹화된 픽스처 없음: {path.name} (scripts/record_sharadar_fixtures.py)")
        payload = json.loads(path.read_text())["payload"]
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows or not isinstance(rows[0], dict):
            pytest.skip(f"{table}: 레코드 형식이 아님")
        return rows

    def test_date_column_is_unified(self) -> None:
        """전 테이블이 기준 날짜를 `date` 로 보낸다는 가정."""
        for table in ("sf1", "sf2", "sf3a"):
            rows = self._load(table)
            assert "date" in rows[0], f"{table}: `date` 컬럼이 사라졌다 — 리네임 규칙 재확인 필요"

    def test_daily_marketcap_is_in_millions(self) -> None:
        """DAILY 의 marketcap 이 여전히 백만 달러 단위인지 (자릿수로 판별)."""
        rows = self._load("daily")
        mcap = float(rows[0]["marketcap"])
        assert 1e3 < mcap < 1e8, (
            f"marketcap={mcap} — 백만 달러 단위 가정에서 벗어났다. "
            "달러로 바뀌었다면 daily_metrics 의 1e6 환산을 제거해야 한다"
        )

    def test_arq_average_balance_columns_are_null(self) -> None:
        """
        ARQ 차원에서 assetsavg/equityavg/invcapavg 가 null 이라는 실측.
        벤더가 채우기 시작하면 avg_balance() 직접 계산을 재검토해야 한다.
        """
        rows = self._load("sf1")
        for col in ("assetsavg", "equityavg", "invcapavg"):
            if col not in rows[0]:
                continue
            values = [r.get(col) for r in rows]
            assert all(v in (None, "", "None") for v in values), (
                f"{col} 이 채워져 있다 — avg_balance() 직접 계산이 불필요해졌을 수 있다"
            )
