from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.taa.backtest import run_backtest
from opt_portfolio.taa.strategy import StrategySpec

STATIC = StrategySpec(
    name="allspy",
    canary=(),
    offensive=(),
    defensive=(),
    top_n_offensive=0,
    top_n_defensive=0,
    static_weights={"SPY": 1.0},
)


def _daily(n_months: int = 40, monthly_step: float = 0.01) -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", periods=n_months * 21, freq="B")
    ramp = (1 + monthly_step) ** (np.arange(len(idx)) / 21)
    return pd.DataFrame({"SPY": 100.0 * ramp, "IEF": 100.0 * np.ones(len(idx))}, index=idx)


class TestRunBacktest:
    def test_static_full_allocation_tracks_the_asset(self) -> None:
        daily = _daily()
        out = run_backtest(STATIC, daily, cost_bps=0.0)

        monthly = daily["SPY"].resample("ME").last()
        expected = monthly.iloc[-1] / monthly.iloc[len(monthly) - len(out.returns) - 1] - 1
        assert (1 + out.returns).prod() - 1 == pytest.approx(expected, rel=1e-6)

    def test_costs_reduce_returns(self) -> None:
        daily = _daily()
        free = run_backtest(STATIC, daily, cost_bps=0.0)
        costly = run_backtest(STATIC, daily, cost_bps=100.0)

        assert costly.equity.iloc[-1] < free.equity.iloc[-1]

    def test_static_allocation_pays_no_turnover_cost_after_entry(self) -> None:
        """비중이 안 바뀌면 회전이 없다 — 비용을 매달 물리면 안 된다."""
        daily = _daily()
        free = run_backtest(STATIC, daily, cost_bps=0.0)
        costly = run_backtest(STATIC, daily, cost_bps=100.0)

        gap = free.equity.iloc[-1] - costly.equity.iloc[-1]
        assert gap < free.equity.iloc[-1] * 0.02

    def test_no_lookahead_signal_uses_prior_month_close(self) -> None:
        """t월말 신호로 t+1월 수익을 얻는다. 같은 달 수익을 쓰면 룩어헤드다."""
        daily = _daily()
        out = run_backtest(STATIC, daily, cost_bps=0.0)
        monthly = daily["SPY"].resample("ME").last()

        assert out.returns.index[-1] == monthly.index[-1]
        assert len(out.returns) < len(monthly)

        # 인덱스 정렬만으로는 `.shift(-1)` 을 지워도(같은 달 수익 사용) 안 걸린다 —
        # 값 자체가 실제 월별 수익률과 일치하는지까지 확인한다.
        expected = monthly.pct_change(fill_method=None)
        pd.testing.assert_series_equal(
            out.returns, expected.reindex(out.returns.index), check_names=False
        )

    def test_equity_and_returns_are_consistent(self) -> None:
        out = run_backtest(STATIC, _daily(), cost_bps=10.0)

        assert out.equity.iloc[-1] == pytest.approx(
            out.equity.iloc[0] * (1 + out.returns).prod(), rel=1e-9
        )

    def test_defensive_ratio_is_zero_for_static(self) -> None:
        assert run_backtest(STATIC, _daily(), cost_bps=0.0).defensive_ratio == 0.0

    def test_mid_series_missing_price_raises_instead_of_silently_skipping(self) -> None:
        """구간 중간의 가격 공백은 조용히 건너뛰면 안 된다 — 예외로 드러나야 한다.

        pandas `pct_change` 의 기본 `fill_method='pad'` 는 결측월을 직전 값으로
        메워 0% 수익으로 둔갑시킨다. 그 상태에서는 이 결측이 절대 감지되지
        않는다 — 이 테스트는 그 절단이 새어나가지 않는지를 확인한다.
        """
        daily = _daily().copy()
        # resample("ME").last() 는 그룹 내 NaN 을 건너뛰고 직전 유효값을 쓴다 —
        # 월말 하루만 지워서는 결측이 재현되지 않는다. 구간 중간 한 달 전체를
        # 지워야 해당 월말가가 실제로 NaN 이 된다.
        months = daily.index.to_period("M")
        gap_month = months.unique()[20]  # 구간 중간의 한 달
        daily.loc[months == gap_month, "SPY"] = np.nan

        with pytest.raises(ValueError, match="수익률 결측"):
            run_backtest(STATIC, daily, cost_bps=0.0)
