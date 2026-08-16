from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.taa.backtest import (
    BacktestOutput,
    _build_tranche_sleeves,
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

#: 신호(SMA13) 로 A/B 중 하나를 고르는 스펙. `SPEC` 은 정적 100% SPY 라
#: 신호 관측일이 언제든 결정이 절대 안 바뀐다 — 트랜치가 존재 이유로 삼는
#: "타이밍에 따라 다른 결정" 자체가 나올 수 없는 퇴화 케이스다. 트랜치의
#: 효과(그리고 그 한계)를 재려면 실제로 신호에 반응하는 스펙이 필요하다.
SIGNAL_SPEC = StrategySpec(
    name="pick",
    canary=(),
    offensive=("A", "B"),
    defensive=("A", "B"),
    top_n_offensive=1,
    top_n_defensive=1,
    selection="sma13",
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


def _dual_asset_noisy(seed: int = 27, noise: float = 5.0) -> pd.DataFrame:
    """A·B 두 자산이 같은 추세를 타지만 잡음 때문에 SMA13 순위가 이따금
    뒤집힌다. `SIGNAL_SPEC` 이 매달 둘 중 하나를 골라 전액 투자하므로,
    순위가 뒤집히는 달에는 신호 관측일(트랜치 오프셋)에 따라 실제로 다른
    자산이 뽑힌다 — 그래야 트랜치의 타이밍-분산 축소 효과를 측정할 신호가
    생긴다.

    `noise=5.0` 은 순위 뒤집힘이 실제로 일어날 만큼 잡음을 주기 위한
    스케일이고, `seed=27` 기본값은 `TestTrancheSleeveCorrelation` 이 상관계수
    행렬 하나를 고정해 보여주는 데 쓴다(그 테스트는 부등호 방향만 보는 게
    아니라 실제 행렬 수치를 리포트해야 하므로 시드 하나가 필요하다). **분산
    축소 자체는 시드 하나로 주장하지 않는다** — 처음엔 이 시드를 그리드
    서치로 골라 "std 0.86배·상관계수 0.7" 이 나오게 맞췄었는데, 그건 데이터를
    주장에 맞춘 것이지 주장을 데이터로 검증한 게 아니었다. `TestTranches`
    쪽은 이제 여러 시드(0~9)를 독립적으로 돌려 방향성을 본다.
    """
    n = 40 * 21
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    base = np.linspace(100, 200, n)
    a = base + rng.normal(0.0, noise, n).cumsum() * 0.05
    b = base + rng.normal(0.0, noise, n).cumsum() * 0.05
    return pd.DataFrame({"A": a, "B": b}, index=idx)


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
    """`SPEC`(정적 100% SPY)은 이 슬리브 테스트들에 못 쓴다 — 정적 배분은
    신호 관측일이 언제든 매달 같은 결정을 내리므로, 어느 슬리브를 골라도
    실현 수익이 완전히 동일하다(트랜치가 plain 과 정확히 같아짐). 그래서
    `SIGNAL_SPEC` + `_dual_asset_noisy` 로, 관측일에 따라 실제로 다른
    자산이 뽑히는 상황을 만든다.
    """

    def test_tranche_returns_have_lower_dispersion_across_many_seeds(self) -> None:
        """트랜치는 수익을 좇는 게 아니라 분산을 줄이는 장치다 — 다만 실제
        구현(모든 슬리브가 같은 달력월의 진짜 가격을 공유)에서는 그 축소폭이
        작다.

        시드 하나로 이 방향성을 주장하면 순환논증이 된다: 그 시드 자체를
        "축소가 뚜렷하게 나오는" 값으로 그리드서치해서 고른 것이면, 테스트는
        "그 시드가 그 시드답게 군다"는 것 이상을 확인하지 못한다(리뷰
        지적). 그래서 서로 독립인 시드 10개(0~9, `noise=5.0` 고정 — 시드별로
        노이즈 스케일까지 바꾸진 않는다)에 대해 각각 `spread/plain` std 비율을
        구하고, **대다수**(8/10 이상)가 1.0 미만인지와 **평균** 비율이
        1.0 미만인지를 본다. 실측(2026-08-17): 9/10 이 비율 < 1.0, 평균 비율
        ≈0.962 — 뚜렷하진 않아도 방향은 일관되게 축소 쪽이다. 이 실측이
        바뀌어 방향성 자체가 깨지면(예: 대다수가 축소를 안 보이면) 이 테스트가
        그대로 실패해야 한다 — 통과시키려고 문턱값을 다시 맞추지 않는다.
        """
        ratios = []
        for seed in range(10):
            daily = _dual_asset_noisy(seed=seed)
            plain = run_backtest(SIGNAL_SPEC, daily, cost_bps=0.0)
            spread = run_with_tranches(SIGNAL_SPEC, daily, cost_bps=0.0)
            ratios.append(float(spread.returns.std() / plain.returns.std()))

        reduced = sum(r < 1.0 for r in ratios)
        assert reduced >= 8, f"10개 시드 중 {reduced}개만 축소를 보였다: {ratios}"
        assert np.mean(ratios) < 1.0, f"평균 비율이 1.0 이상이다: {ratios}"

    def test_tranche_returns_differ_from_any_single_tranche(self) -> None:
        """평균이 트랜치 하나(예: 오프셋 0)로 몰래 대체돼도 std 상한만으론
        못 잡을 수 있다 — 오프셋 0 은 plain 과 정확히 같아서다. 값 자체를
        직접 대조해 "평균"이 실제로 4개를 섞었는지 확인한다.
        """
        daily = _dual_asset_noisy()
        plain = run_backtest(SIGNAL_SPEC, daily, cost_bps=0.0)
        spread = run_with_tranches(SIGNAL_SPEC, daily, cost_bps=0.0)

        common = spread.returns.index.intersection(plain.returns.index)
        assert not np.allclose(spread.returns.reindex(common), plain.returns.reindex(common))

    def test_tranche_output_has_same_index_as_plain(self) -> None:
        daily = _crash_then_recover()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        spread = run_with_tranches(SPEC, daily, cost_bps=0.0)

        assert spread.returns.index.equals(plain.returns.index)


class TestTrancheSleeveCorrelation:
    """슬리브가 진짜 트랜치(같은 시장을 다른 관측일로 보는 것)인지, 아니면
    그냥 서로 다른 기간의 수익률을 평균 낸 스무딩인지를 가른다.

    네 슬리브는 같은 전략을 같은 자산에, 겨우 며칠 어긋난 관측일로 돌린다
    — 상관관계가 낮을 이유가 없다. 예전 구현(전체 가격 패널을 통째로
    `shift`)은 슬리브 0↔3 이 실측 상관계수 0.076 까지 떨어졌다 — 사실상
    무관한 두 시계열처럼 보였다는 뜻이고, 그게 인위적인 분산 축소(따라서
    부풀려진 Calmar·PBO)의 원인이었다. 이 테스트가 없으면 그 착시가 다시
    "성공"으로 통과한다.
    """

    def test_sleeve_pairwise_correlations_stay_high(self) -> None:
        daily = _dual_asset_noisy()
        sleeves = _build_tranche_sleeves(
            SIGNAL_SPEC,
            daily,
            n_tranches=4,
            ma_overlay=False,
            benchmark="SPY",
            ma_days=200,
            start=None,
            end=None,
            cost_bps=0.0,
        )
        matrix = pd.DataFrame({i: s.returns for i, s in enumerate(sleeves)}).dropna(how="any")
        corr = matrix.corr().to_numpy()
        off_diagonal = corr[~np.eye(len(sleeves), dtype=bool)]

        assert off_diagonal.min() > 0.7


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
