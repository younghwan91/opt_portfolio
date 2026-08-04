"""공용 픽스처 — 합성 벤더 데이터로 채운 PIT 스토어."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.data.store import PITStore

START, END = "2018-01-02", "2021-12-31"
QUARTERS = pd.date_range("2017-03-31", "2021-12-31", freq="QE")

#: 일반 종목 20개 + 특수 케이스
NORMAL = [f"T{i:02d}" for i in range(20)]
SPECIAL = {
    "FINCO": {"sector": "Financial Services", "industry": "Banks - Regional"},
    "CHINACO": {"location": "Beijing, China"},
    "OILLP": {"name": "Oil Energy Partners L.P."},
    "PENNY": {},  # 주가 $2
    "LOSSCO": {},  # 만년 적자
}
TICKERS = NORMAL + list(SPECIAL)
BENCH = "SPY"


def populate(store: PITStore, seed: int = 21) -> None:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="B")
    all_tickers = TICKERS + [BENCH]

    # ---- 가격: 종목별 드리프트 = 품질 순위와 연동 (팩터에 진짜 신호 부여)
    quality = np.linspace(0.0006, -0.0002, len(TICKERS))
    drift = np.append(quality, 0.0003)  # SPY
    rets = rng.normal(drift, 0.018, (len(days), len(all_tickers)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=days, columns=all_tickers)
    close["PENNY"] = 2.0
    volume = pd.DataFrame(1_000_000.0, index=days, columns=all_tickers)
    mcap = close * 10_000_000.0

    prices = pd.concat(
        [
            close.stack().rename("close"),
            close.stack().rename("closeunadj"),
            volume.stack().rename("volume"),
            mcap.stack().rename("mcap"),
        ],
        axis=1,
    ).reset_index()
    prices.columns = ["date", "ticker", "close", "closeunadj", "volume", "mcap"]
    store.upsert_prices(prices)

    # ---- 분기 재무 (SPY 제외): datekey = 기간말 + 45일
    rows = []
    for i, ticker in enumerate(TICKERS):
        base_income = 50.0 + 20.0 * quality[min(i, len(quality) - 1)] * 1e4
        for q_idx, q in enumerate(QUARTERS):
            netinc = base_income * (1.02**q_idx)
            if ticker == "LOSSCO":
                netinc = -10.0
            rows.append(
                {
                    "ticker": ticker,
                    "calendardate": q,
                    "datekey": q + pd.Timedelta(days=45),
                    "dimension": "ARQ",
                    "revenue": abs(netinc) * 5.0,
                    "netinc": netinc,
                    "gp": abs(netinc) * 2.5,
                    "assets": 1000.0 + 10.0 * q_idx,
                    "equity": 500.0,
                    "ncfo": netinc * 1.1,
                    "sharesbas": 1_000_000.0,
                }
            )
    store.upsert_fundamentals(pd.DataFrame(rows))

    # ---- 13F 집계: datekey = 기간말 + 75일 (SF1 보다 늦게 — avail 분리 검증용)
    own_rows = [
        {
            "ticker": t,
            "calendardate": q,
            "datekey": q + pd.Timedelta(days=75),
            "inst_shares": 400_000.0 + 1_000.0 * i,
            "inst_holders": 50.0 + i,
        }
        for i, q in enumerate(QUARTERS)
        for t in NORMAL[:5]
    ]
    store.upsert_institutions(pd.DataFrame(own_rows))

    # ---- 메타
    meta_rows = []
    for t in all_tickers:
        info = SPECIAL.get(t, {})
        meta_rows.append(
            {
                "ticker": t,
                "sector": info.get("sector", "Technology"),
                "industry": info.get("industry", "Software - Application"),
                "location": info.get("location", "California; U.S.A"),
                "category": "Domestic Common Stock" if t != BENCH else "ETF",
                "name": info.get("name", f"{t} Inc"),
                "siccode": "7372",
                "is_delisted": "N",
            }
        )
    store.upsert_tickers(pd.DataFrame(meta_rows))


@pytest.fixture(scope="module")
def synth_store() -> PITStore:
    store = PITStore(":memory:")
    populate(store)
    yield store
    store.close()


@pytest.fixture(scope="module")
def synth_ctx(synth_store: PITStore):
    return synth_store.build_context(benchmark=BENCH)
