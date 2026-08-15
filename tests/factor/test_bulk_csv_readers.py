"""벌크 CSV 리더가 **실제 벤더 파일의 컬럼 이름**을 받는지.

이 파일이 생긴 이유: `_csv_institutions` 가 `calendardate` 를 기대했는데
`holdings_ticker.csv` 의 실제 헤더는 `date` 였다. API 경로는
`_DIRECT_RENAME["SF3A"]` 가 리네임해 주지만 CSV 경로에는 그게 없어서, 이
리더는 **한 번도 동작한 적이 없었다** (2026-08-15 전량 재구축 중 `KeyError`
로 발견). 스토어의 `institutions` 가 0행이었던 원인이다.

아래 헤더들은 2026-08-12 자 벌크 zip 에서 그대로 뜬 것이다 — 추측이 아니다.
"""

from __future__ import annotations

import pandas as pd

from opt_portfolio.factor.data.sharadar import (
    _csv_daily,
    _csv_institutions,
    _csv_tickers,
)


def test_institutions_accepts_the_bulk_columns_date_not_calendardate():
    """holdings_ticker.csv 실측 헤더: date,ticker,name,shrholders,…,shrunits,…"""
    raw = pd.DataFrame(
        {
            "date": ["2026-06-30"],
            "ticker": ["AAPL"],
            "name": ["Apple Inc"],
            "shrholders": [4321],
            "shrunits": [9_876_543],
        }
    )

    out = _csv_institutions(raw)

    assert out["calendardate"].iloc[0] == "2026-06-30"
    assert out["inst_holders"].iloc[0] == 4321
    assert out["inst_shares"].iloc[0] == 9_876_543


def test_institutions_datekey_is_quarter_end_plus_the_13f_deadline():
    """13F 는 분기말 +45일이 법정 기한 — 그 전엔 시장이 못 본 정보다."""
    raw = pd.DataFrame({"date": ["2026-06-30"], "ticker": ["AAPL"], "shrunits": [1]})

    out = _csv_institutions(raw)

    assert out["datekey"].iloc[0] == pd.Timestamp("2026-08-14")  # 6/30 + 45일


def test_institutions_still_accepts_the_api_column_name():
    """API 경로는 이미 calendardate 로 넘긴다 — 그쪽을 깨뜨리면 안 된다."""
    raw = pd.DataFrame({"calendardate": ["2026-06-30"], "ticker": ["AAPL"], "shrunits": [1]})

    out = _csv_institutions(raw)

    assert out["calendardate"].iloc[0] == "2026-06-30"


def test_daily_csv_converts_marketcap_from_millions():
    """daily.csv 의 marketcap/ev 는 백만 달러 단위 — 미환산 시 배수가 10⁶배 틀어진다."""
    raw = pd.DataFrame(
        {"ticker": ["AAPL"], "date": ["2026-08-14"], "marketcap": [3_500_000.0], "ev": [3_400_000.0]}
    )

    out = _csv_daily(raw)

    assert out["mcap"].iloc[0] == 3.5e12
    assert out["ev"].iloc[0] == 3.4e12


def test_tickers_csv_keeps_the_delisted_flag_reachable():
    """CSV 는 isdelisted, 스토어는 is_delisted — 리네임이 빠지면 폐지 여부가 통째로 NULL."""
    raw = pd.DataFrame(
        {"ticker": ["ENRN"], "name": ["Enron"], "isdelisted": ["Y"], "sector": ["Energy"]}
    )

    out = _csv_tickers(raw)

    assert "is_delisted" in out.columns
    assert out["is_delisted"].iloc[0] == "Y"
