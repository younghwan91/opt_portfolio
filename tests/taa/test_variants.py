from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.taa.backtest import (
    BacktestOutput,
    run_backtest,
    run_with_ma_overlay,
    run_with_tranches,
)
from opt_portfolio.taa.strategy import StrategySpec

RunFn = Callable[..., BacktestOutput]

SPEC = StrategySpec(
    name="allspy",
    canary=(),
    offensive=(),
    defensive=(),
    top_n_offensive=0,
    top_n_defensive=0,
    static_weights={"SPY": 1.0},
)


def _crash_then_recover() -> pd.DataFrame:
    """앞 절반 상승, 뒤 절반 급락 — 이평 오버레이가 뒤쪽을 잘라야 한다."""
    n = 40 * 21
    up = np.linspace(100, 200, n // 2)
    down = np.linspace(200, 90, n - n // 2)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    px = np.concatenate([up, down])
    return pd.DataFrame({"SPY": px, "IEF": np.full(n, 100.0)}, index=idx)


def _first_month_loss() -> pd.DataFrame:
    """13개월 상승(모멘텀이 usable 해지는 최소 이력) → 1개월 급락 → 그 후로는
    쭉 상승, 낙폭을 다시 갱신하지 않는다.

    낙폭이 처음이자 마지막이라는 게 핵심이다. 이후 구간에 더 큰 하락이 없으면
    전체 시계열의 MDD 는 오직 이 첫 달 손실에서만 나온다. `equity` 가
    `run_backtest` 규약대로 진입 시점의 원금 10,000 을 맨 앞에 붙이지 않으면,
    첫 손실을 담을 cummax 기준점 자체가 없어(첫 원소가 곧 자기 자신의 cummax)
    MDD 가 0 으로 완전히 사라진다 — 이후 구간의 큰 폭락이 우연히 같은 값을
    가려버리는 걸 막기 위해 일부러 이렇게 설계했다.
    """
    up1_months, dip_months, up2_months = 13, 1, 26
    up1 = np.linspace(100, 150, up1_months * 21)
    dip = np.linspace(150, 110, dip_months * 21)
    up2 = np.linspace(110, 400, up2_months * 21)
    idx = pd.date_range("2010-01-01", periods=len(up1) + len(dip) + len(up2), freq="B")
    px = np.concatenate([up1, dip, up2])
    return pd.DataFrame({"SPY": px, "IEF": np.full(len(idx), 100.0)}, index=idx)


def _crash_then_recover_noisy() -> pd.DataFrame:
    """`_crash_then_recover` + 일별 잡음.

    순수 결정론적 램프에서는 모든 트랜치가 (오프셋만 다를 뿐) 같은 모양의
    경로를 보고 같은 월말 수익을 뽑아낸다 — 분산 축소 효과를 측정할 신호
    자체가 없다. 잡음을 얹어야 "어느 요일에 리밸런싱했는지"가 실제로
    갈리고, 트랜치 평균이 분산을 줄인다는 주장을 통계적으로 검증할 수 있다.
    시드 고정으로 재현 가능하다.
    """
    daily = _crash_then_recover()
    rng = np.random.default_rng(42)
    noisy = daily.copy()
    noisy["SPY"] = daily["SPY"] + rng.normal(0.0, 1.5, len(daily))
    return noisy


class TestMaOverlay:
    def test_overlay_reduces_drawdown_in_a_crash(self) -> None:
        daily = _crash_then_recover()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        overlaid = run_with_ma_overlay(SPEC, daily, cost_bps=0.0)

        def mdd(eq: pd.Series) -> float:
            return float((eq / eq.cummax() - 1).min())

        assert mdd(overlaid.equity) > mdd(plain.equity)

    def test_overlay_is_flat_when_always_above_ma(self) -> None:
        n = 40 * 21
        idx = pd.date_range("2010-01-01", periods=n, freq="B")
        daily = pd.DataFrame({"SPY": np.linspace(100, 300, n), "IEF": np.full(n, 100.0)}, index=idx)
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        overlaid = run_with_ma_overlay(SPEC, daily, cost_bps=0.0)

        pd.testing.assert_series_equal(plain.returns, overlaid.returns)


class TestTranches:
    def test_tranche_returns_have_lower_dispersion(self) -> None:
        """트랜치는 수익을 좇는 게 아니라 분산을 줄이는 장치다.

        결정론적 램프(`_crash_then_recover`)에서는 모든 오프셋이 같은 모양의
        경로를 보므로 분산 축소가 통계적으로 드러나지 않는다 — 잡음을 얹은
        `_crash_then_recover_noisy` 로 실제 마진(≈0.84배, 사전 실측)을 확인한다.
        1.01배 같은 느슨한 상한은 "트랜치 0 만 반환"하는 고장에도 통과하므로
        (오프셋 0 == plain 이라 정확히 1.0배) 실질적 개선을 요구하는 0.95배로 조인다.
        """
        daily = _crash_then_recover_noisy()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        spread = run_with_tranches(SPEC, daily, cost_bps=0.0)

        assert spread.returns.std() < plain.returns.std() * 0.95

    def test_tranche_returns_differ_from_any_single_tranche(self) -> None:
        """평균이 트랜치 하나(예: 오프셋 0)로 몰래 대체돼도 std 상한만으론
        못 잡을 수 있다 — 오프셋 0 은 plain 과 정확히 같아서다. 값 자체를
        직접 대조해 "평균"이 실제로 4개를 섞었는지 확인한다.
        """
        daily = _crash_then_recover_noisy()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        spread = run_with_tranches(SPEC, daily, cost_bps=0.0)

        common = spread.returns.index.intersection(plain.returns.index)
        assert not np.allclose(spread.returns.reindex(common), plain.returns.reindex(common))

    def test_tranche_output_has_same_index_as_plain(self) -> None:
        daily = _crash_then_recover()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        spread = run_with_tranches(SPEC, daily, cost_bps=0.0)

        assert spread.returns.index.equals(plain.returns.index)


class TestComposition:
    def test_tranches_with_ma_overlay_differs_from_either_alone(self) -> None:
        """`baa_bal_ma_tranche` 처럼 두 변형을 겹쳐 쓰는 구성이 실제로 존재한다
        (`registry.MA_OVERLAY` 와 `registry.TRANCHE` 양쪽에 다 들어있다). 겹쳐
        쓰기가 조용히 트랜치만 적용하고 오버레이를 빼먹으면 (또는 그 반대면)
        구성 9번이 구성 8번의 숫자를 그대로 복제하게 된다 — 이게 실측으로
        재현된 회귀였다. 합성 결과가 두 단일 변형 중 어느 쪽과도 같지 않아야
        진짜로 둘 다 반영됐다고 볼 수 있다.
        """
        daily = _crash_then_recover()
        tranche_only = run_with_tranches(SPEC, daily, cost_bps=0.0)
        overlay_only = run_with_ma_overlay(SPEC, daily, cost_bps=0.0)
        composed = run_with_tranches(SPEC, daily, ma_overlay=True, cost_bps=0.0)

        c1 = tranche_only.returns.index.intersection(composed.returns.index)
        c2 = overlay_only.returns.index.intersection(composed.returns.index)
        assert not np.allclose(tranche_only.returns.reindex(c1), composed.returns.reindex(c1))
        assert not np.allclose(overlay_only.returns.reindex(c2), composed.returns.reindex(c2))


VARIANTS: list[RunFn] = [run_backtest, run_with_ma_overlay, run_with_tranches]
VARIANT_IDS = ["plain", "ma_overlay", "tranches"]


class TestEquityConvention:
    """`BacktestOutput.equity` 규약(`run_backtest` 가 세운 것 — `04-data-contract.md`
    아님, `backtest.py` 의 docstring)이 세 진입점 모두에서 지켜지는지 확인한다.

    이걸 하나씩 따로 쓰지 않고 파라미터화한 이유: 앞서 `run_with_ma_overlay`,
    `run_with_tranches` 둘 다 각각 `equity=(1 + returns).cumprod() * 10_000.0`
    로 진입 원금 없이 만들었었다 — `run_backtest` 만 테스트하고 두 변형은
    안 했기 때문에 이 회귀가 통과했다. 세 진입점을 한 목록으로 돌리면 앞으로
    네 번째 변형이 추가돼도 이 목록에 넣는 것만으로 같은 검증을 받는다.
    """

    @pytest.mark.parametrize("run_fn", VARIANTS, ids=VARIANT_IDS)
    def test_equity_has_one_more_point_than_returns(self, run_fn: RunFn) -> None:
        daily = _crash_then_recover()
        out = run_fn(SPEC, daily, cost_bps=0.0)

        assert len(out.equity) == len(out.returns) + 1

    @pytest.mark.parametrize("run_fn", VARIANTS, ids=VARIANT_IDS)
    def test_equity_last_point_matches_compounded_returns(self, run_fn: RunFn) -> None:
        daily = _crash_then_recover()
        out = run_fn(SPEC, daily, cost_bps=0.0)

        expected = out.equity.iloc[0] * float((1 + out.returns).prod())
        assert out.equity.iloc[-1] == pytest.approx(expected)

    @pytest.mark.parametrize("run_fn", VARIANTS, ids=VARIANT_IDS)
    def test_first_month_loss_shows_up_in_drawdown_from_equity(self, run_fn: RunFn) -> None:
        """첫 usable 달의 손실이 `equity` 기반 MDD 에 반영돼야 한다. `_first_month_loss`
        는 그 손실 이후로 낙폭을 다시 갱신하지 않으므로, 진입 원금 포인트가
        빠지면(과거 버그) MDD 가 정확히 0 으로 사라진다 — 실측: 세 진입점 모두
        원금 포함 시 음수(plain/overlay -15.1%, tranches -1.8%), 원금 누락 시 0.0%.
        """
        daily = _first_month_loss()
        out = run_fn(SPEC, daily, cost_bps=0.0)

        mdd = float((out.equity / out.equity.cummax() - 1).min())
        assert mdd < -0.01
