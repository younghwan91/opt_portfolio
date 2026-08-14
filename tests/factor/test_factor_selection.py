"""
학습 구간 내 팩터 선택 — 조합 탐색을 정직하게 만드는 장치.

지금까지는 **전 구간 IC 를 보고 사람이 팩터를 골랐다.** 그 선택은 DSR 의
시도 횟수에 들어가 있지 않으므로 미정산 부채다. 팩터 158개 중 5개를
고르는 조합은 8억 가지이고, 그중 최고를 사후에 고르면 백테스트는 반드시
좋아진다.

해법은 선택을 **학습 구간 안으로** 옮기는 것이다. 각 폴드에서 학습 데이터로만
팩터를 고르고, 그 조합으로 검증 구간을 한 번 실행한다. 그러면 조합 탐색
자체가 OOS 로 검증된다.

이 파일이 지키는 불변식:
  ① 선택은 지정 구간 밖 데이터를 절대 보지 않는다
  ② 순방향 수익의 확정 지연을 감안한다 (IC 는 t+horizon 이 지나야 안다)
  ③ 근거가 부족하면 조용히 이상한 걸 고르지 않고 폴백한다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.research.selection import select_factors


@pytest.fixture
def panels_and_returns():
    """A 는 진짜 신호, B 는 노이즈, C 는 후반부에만 작동한다."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2000-01-31", periods=180, freq="ME")
    tickers = [f"T{i:02d}" for i in range(40)]
    fwd = pd.DataFrame(
        rng.normal(0.01, 0.06, (len(dates), len(tickers))), index=dates, columns=tickers
    )
    noise = lambda s: pd.DataFrame(  # noqa: E731
        rng.normal(size=fwd.shape) * s, index=dates, columns=fwd.columns
    )
    panels = {
        "GOOD": fwd + noise(0.04),  # 순방향 수익과 강하게 연동
        "NOISE": noise(1.0),
        "LATE": pd.concat([noise(1.0).iloc[:120], (fwd + noise(0.04)).iloc[120:]]),
    }
    return panels, fwd


class TestSelectFactors:
    def test_picks_the_predictive_factor(self, panels_and_returns) -> None:
        panels, fwd = panels_and_returns

        picked = select_factors(panels, fwd, end=fwd.index[-1], k=1)

        assert list(picked) == ["GOOD"]

    def test_rejects_pure_noise_when_room_remains(self, panels_and_returns) -> None:
        """자리가 남아도 IC 가 0 이하인 팩터는 담지 않는다."""
        panels, fwd = panels_and_returns

        picked = select_factors(panels, fwd, end=fwd.index[-1], k=3, min_ic=0.02)

        assert "NOISE" not in picked

    def test_selection_uses_only_data_before_end(self, panels_and_returns) -> None:
        """
        핵심 불변식. LATE 는 120개월 이후에만 작동하므로, 그 이전을 기준으로
        고르면 선택되면 안 된다. 선택되면 미래를 본 것이다.
        """
        panels, fwd = panels_and_returns

        early = select_factors(panels, fwd, end=fwd.index[100], k=2, min_ic=0.02)

        assert "LATE" not in early, "학습 구간 밖 데이터를 봤다"

    def test_late_factor_is_picked_once_it_works(self, panels_and_returns) -> None:
        panels, fwd = panels_and_returns

        late = select_factors(panels, fwd, end=fwd.index[-1], k=2, min_ic=0.02)

        assert "LATE" in late

    def test_horizon_lag_shrinks_usable_history(self, panels_and_returns) -> None:
        """
        IC 는 순방향 수익을 쓰므로 t 시점의 IC 는 t+horizon 이 지나야 안다.
        지연이 크면 쓸 수 있는 관측이 줄고, min_months 에 못 미치면
        선택을 포기하고 전체로 후퇴한다 — 그 경계가 관측 가능한 결과다.
        """
        panels, fwd = panels_and_returns
        end = fwd.index[40]  # 41개월치

        selective = select_factors(panels, fwd, end=end, k=1, min_ic=0.0, min_months=36, lag=1)
        starved = select_factors(panels, fwd, end=end, k=1, min_ic=0.0, min_months=36, lag=12)

        assert len(selective) == 1, "관측이 충분하면 골라야 한다"
        assert set(starved) == set(panels), "관측이 부족하면 전체로 후퇴해야 한다"

    def test_insufficient_history_falls_back_to_all(self, panels_and_returns) -> None:
        """관측이 부족하면 임의로 고르지 않고 전체를 쓴다 — 1/N 으로 후퇴."""
        panels, fwd = panels_and_returns

        picked = select_factors(panels, fwd, end=fwd.index[3], k=2, min_months=36)

        assert set(picked) == set(panels)

    def test_k_larger_than_pool_is_safe(self, panels_and_returns) -> None:
        panels, fwd = panels_and_returns

        picked = select_factors(panels, fwd, end=fwd.index[-1], k=99, min_ic=-99)

        assert set(picked) == set(panels)
