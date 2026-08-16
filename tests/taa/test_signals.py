from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.taa.signals import momentum_13612w, sma_ratio, to_monthly


def _monthly(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    return pd.DataFrame({"A": start + step * np.arange(n)}, index=idx)


class TestToMonthly:
    def test_takes_last_observation_of_month(self) -> None:
        idx = pd.date_range("2010-01-01", periods=45, freq="D")
        daily = pd.DataFrame({"A": np.arange(45, dtype=float)}, index=idx)

        m = to_monthly(daily)

        assert m.loc[pd.Timestamp("2010-01-31"), "A"] == 30.0


class TestMomentum13612W:
    def test_matches_hand_computation(self) -> None:
        """12*r1 + 4*r3 + 2*r6 + 1*r12 — 논문 정의 그대로."""
        m = _monthly(13)
        mom = momentum_13612w(m)

        p = m["A"]
        expected = (
            12 * (p.iloc[12] / p.iloc[11] - 1)
            + 4 * (p.iloc[12] / p.iloc[9] - 1)
            + 2 * (p.iloc[12] / p.iloc[6] - 1)
            + 1 * (p.iloc[12] / p.iloc[0] - 1)
        )
        assert mom["A"].iloc[-1] == pytest.approx(expected)

    def test_needs_twelve_months_of_history(self) -> None:
        """12개월이 안 차면 NaN — 없는 데이터로 판정하지 않는다."""
        mom = momentum_13612w(_monthly(12))

        assert mom["A"].isna().all()

    def test_negative_when_price_falls(self) -> None:
        m = _monthly(13, start=200.0, step=-5.0)

        assert momentum_13612w(m)["A"].iloc[-1] < 0

    def test_missing_month_propagates_nan_not_zero_return(self) -> None:
        """`pct_change` 기본값(fill_method='pad')은 결측월을 직전 값으로 메워
        0% 수익으로 둔갑시킨다 — 결측은 NaN 으로 남아야 모멘텀 점수가 그
        결측을 '무변동'으로 오인하지 않는다."""
        m = _monthly(13)
        m.iloc[6, 0] = np.nan  # 구간 중간 한 달 결측 — 마지막 시점의 6개월 항 분모다

        mom = momentum_13612w(m)

        # 마지막 시점의 6개월 항(price[12]/price[6]-1)이 결측을 분모로 참조한다.
        # 결측이 조용히 직전 값으로 메워지면(fill_method='pad') 이 항이 유한값이
        # 되어 score 전체도 NaN 이 아니라 유한값으로 새어나간다.
        assert not np.isfinite(mom["A"].iloc[-1])


class TestSmaRatio:
    def test_is_price_over_trailing_average(self) -> None:
        m = _monthly(13)
        ratio = sma_ratio(m, window=13)

        p = m["A"]
        assert ratio["A"].iloc[-1] == pytest.approx(p.iloc[-1] / p.iloc[:13].mean())

    def test_above_one_in_uptrend_below_in_downtrend(self) -> None:
        assert sma_ratio(_monthly(13), 13)["A"].iloc[-1] > 1.0
        assert sma_ratio(_monthly(13, 200.0, -5.0), 13)["A"].iloc[-1] < 1.0
