"""
변동성 타게팅 — 이진 타이밍의 연속판.

지금 오버레이는 켜짐/꺼짐 둘뿐이다. 그 하나가 이 저장소 최대의 개선을
냈지만(낙폭 −63.8% → −23.7%), 이진이라 두 가지를 못 한다:
  ① 변동성이 서서히 오르는 구간에서 점진적으로 줄이지 못한다
  ② 변동성이 낮은 구간에서 노출을 늘리지 못한다

Moreira & Muir(2017)는 실현변동성에 반비례해 노출을 조절하면 Sharpe 가
개선된다고 보고한다. 핵심 조건은 **그 시점까지의 변동성만 쓰는 것**이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.backtest.timing import volatility_target_exposure


def _returns(vol: float, n: int = 800, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0003, vol / np.sqrt(252), n), index=idx)


class TestVolatilityTarget:
    def test_calm_market_gets_more_exposure_than_turbulent(self) -> None:
        calm = volatility_target_exposure(_returns(0.08), target_vol=0.15)
        wild = volatility_target_exposure(_returns(0.40), target_vol=0.15)

        assert calm.iloc[-1] > wild.iloc[-1]

    def test_exposure_tracks_target(self) -> None:
        """실현변동성이 목표와 같으면 노출은 1 근처여야 한다."""
        exposure = volatility_target_exposure(_returns(0.15), target_vol=0.15)

        assert 0.7 < exposure.iloc[-1] < 1.4

    def test_leverage_is_capped(self) -> None:
        """변동성이 0 에 가까우면 노출이 발산한다 — 상한이 없으면 위험하다."""
        exposure = volatility_target_exposure(_returns(0.01), target_vol=0.15, max_exposure=1.5)

        assert exposure.max() <= 1.5 + 1e-9

    def test_uses_only_past_data(self) -> None:
        """
        핵심 불변식. 뒤에 폭락 구간을 덧붙여도 앞 구간 노출이 바뀌면 안 된다.
        바뀌면 미래 변동성을 보고 포지션을 정한 것이다.
        """
        base = _returns(0.15, n=800, seed=7)
        crash = pd.Series(
            np.random.default_rng(9).normal(-0.004, 0.05, 200),
            index=pd.date_range(base.index[-1] + pd.Timedelta(days=1), periods=200, freq="B"),
        )

        before = volatility_target_exposure(base, target_vol=0.15)
        after = volatility_target_exposure(pd.concat([base, crash]), target_vol=0.15)

        pd.testing.assert_series_equal(before, after.loc[before.index], check_names=False)

    def test_warmup_is_neutral_not_zero(self) -> None:
        """관측이 모자란 초기 구간에서 0 을 주면 그 기간이 통째로 사라진다."""
        exposure = volatility_target_exposure(_returns(0.15, n=300), window=252)

        assert exposure.iloc[0] == pytest.approx(1.0)
        assert exposure.notna().all()

    def test_floor_prevents_total_exit(self) -> None:
        """
        이진 타이밍과 다른 점 — 완전 청산하지 않는다. 변동성 급등 구간에서
        전량 현금으로 가면 반등을 통째로 놓친다 (2009년에 실측됨).
        """
        exposure = volatility_target_exposure(_returns(1.2), target_vol=0.15, min_exposure=0.2)

        assert exposure.min() >= 0.2 - 1e-9
