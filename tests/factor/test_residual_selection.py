"""
잔차 기여도 선별 — 개별 IC 가 못 보는 것.

실측 동기: 개별 IC 상위 6개를 고르는 방식은 19개 폴드 전부에서 가치·품질
팩터만 뽑았고 성장 팩터 4종은 **한 번도** 선택되지 않았다. 그런데 성장을
포함한 고정 조합이 더 좋았다 (CAGR 16.90% vs 15.83%).

개별 IC 는 팩터를 하나씩 세워놓고 잰다. 서로 비슷한 종목을 고르는 팩터들은
IC 가 높아도 새로 더하는 정보가 적은데, IC 순으로 자르면 그것들이 자리를
다 차지한다. 이 파일이 지키는 것은 그 구분이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.research.selection import (
    SELECTORS,
    select_factors,
    select_factors_residual,
)

N_DATES, N_TICKERS = 60, 80


@pytest.fixture
def panels_and_returns():
    """
    설계: `strong` 이 수익을 가장 잘 맞히고, `twin` 은 그 복제(잡음만 다름),
    `other` 는 IC 는 낮지만 **독립적인** 축이다.

    개별 IC 순서는 strong > twin > other 이므로 IC 기준은 twin 을 고른다.
    잔차 기준은 twin 의 기여도가 0 에 가깝다는 것을 보고 other 를 골라야 한다.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-31", periods=N_DATES, freq="ME")
    cols = [f"T{i:03d}" for i in range(N_TICKERS)]

    def frame(values):
        return pd.DataFrame(values, index=dates, columns=cols)

    signal = rng.normal(size=(N_DATES, N_TICKERS))
    independent = rng.normal(size=(N_DATES, N_TICKERS))
    noise = rng.normal(size=(N_DATES, N_TICKERS))

    fwd = frame(0.9 * signal + 0.5 * independent + 0.6 * noise)
    panels = {
        "strong": frame(signal),
        "twin": frame(signal + 0.05 * rng.normal(size=(N_DATES, N_TICKERS))),
        "other": frame(independent),
    }
    return panels, fwd, dates


class TestResidualSelection:
    def test_ic_selection_picks_the_duplicate(self, panels_and_returns) -> None:
        """기준선 — 개별 IC 는 복제 팩터를 거른다는 개념이 없다."""
        panels, fwd, dates = panels_and_returns

        picked = select_factors(panels, fwd, end=dates[-1], k=2, lag=0, min_months=10)

        assert set(picked) == {"strong", "twin"}

    def test_residual_selection_prefers_the_independent_axis(self, panels_and_returns) -> None:
        """이 함수의 존재 이유. 겹치는 것 대신 새 정보를 더하는 것을 고른다."""
        panels, fwd, dates = panels_and_returns

        picked = select_factors_residual(panels, fwd, end=dates[-1], k=2, lag=0, min_months=10)

        assert picked[0] in {"strong", "twin"}, "첫 팩터는 개별 IC 최대여야 한다"
        assert picked[1] == "other", f"복제 대신 독립 축을 골라야 한다 — 실제 {picked}"

    def test_does_not_look_past_end(self, panels_and_returns) -> None:
        """
        핵심 불변식. 학습 구간 끝 이후를 보면 조합 탐색이 룩어헤드가 된다.
        뒷부분 데이터를 뒤집어도 선택이 바뀌면 안 된다.
        """
        panels, fwd, dates = panels_and_returns
        cut = dates[N_DATES // 2]

        before = select_factors_residual(panels, fwd, end=cut, k=2, lag=0, min_months=10)
        tampered = fwd.copy()
        tampered.loc[tampered.index > cut] *= -1.0
        after = select_factors_residual(panels, tampered, end=cut, k=2, lag=0, min_months=10)

        assert before == after

    def test_falls_back_to_all_when_history_is_short(self, panels_and_returns) -> None:
        """근거가 없을 때 돌아갈 곳은 1/N 이다 (DeMiguel et al. 2009)."""
        panels, fwd, dates = panels_and_returns

        picked = select_factors_residual(panels, fwd, end=dates[3], k=2, lag=0, min_months=36)

        assert set(picked) == set(panels)

    def test_registry_exposes_both_methods(self) -> None:
        """설정 파일의 `select_method` 오타는 조용히 기본값으로 후퇴하면 안 된다."""
        assert SELECTORS["ic"] is select_factors
        assert SELECTORS["residual"] is select_factors_residual
        with pytest.raises(KeyError):
            SELECTORS["typo"]
