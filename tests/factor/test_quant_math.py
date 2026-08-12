"""수학 코어 검증 — research(IC/분위수/DSR/PBO) + portfolio(공분산/비중/합성)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.analysis.metrics import calculate_sharpe_ratio
from opt_portfolio.config import RISK_FREE_RATE
from opt_portfolio.factor.backtest.engine import BacktestResult
from opt_portfolio.factor.optimize.walkforward import annualized_sharpe
from opt_portfolio.factor.portfolio.covariance import ledoit_wolf_cc
from opt_portfolio.factor.portfolio.score import composite_score, rank_normalize
from opt_portfolio.factor.portfolio.weights import (
    black_litterman,
    cap_and_normalize,
    hrp,
    mvo,
    risk_parity,
)
from opt_portfolio.factor.research.ic import forward_returns, rank_ic, summarize_ic
from opt_portfolio.factor.research.overfitting import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from opt_portfolio.factor.research.quantiles import analyze_quantiles

RNG = np.random.default_rng(7)
DATES = pd.date_range("2020-01-01", periods=252, freq="B")
TICKERS = [f"T{i:02d}" for i in range(40)]


def _random_close() -> pd.DataFrame:
    rets = RNG.normal(0.0003, 0.02, (len(DATES), len(TICKERS)))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=DATES, columns=TICKERS)


class TestIC:
    def test_perfect_factor_has_ic_one(self) -> None:
        close = _random_close()
        fwd = forward_returns(close, horizon=5)
        ic = rank_ic(fwd, fwd)  # 스코어 = 순방향 수익 그 자체
        assert ic.dropna().min() == pytest.approx(1.0)

    def test_noise_factor_has_ic_near_zero(self) -> None:
        close = _random_close()
        fwd = forward_returns(close, horizon=5)
        noise = pd.DataFrame(RNG.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
        summary = summarize_ic(rank_ic(noise, fwd), horizon=5)
        assert abs(summary.mean) < 0.05
        assert not summary.is_significant()

    def test_forward_return_excludes_signal_day(self) -> None:
        """신호일 t 의 수익률은 fwd 에 포함되면 안 된다 (t+1 진입)."""
        close = pd.DataFrame(
            {"A": [100.0, 100, 200, 200, 200, 200]},
            index=pd.date_range("2024-01-01", periods=6, freq="B"),
        )
        fwd = forward_returns(close, horizon=1)
        # 신호 t=1 (점프 직전): 진입 t=2 (점프 후) → 수익 0
        assert fwd["A"].iloc[1] == pytest.approx(0.0)
        # 신호 t=0: 진입 t=1 (100), 청산 t=2 (200) → +100%
        assert fwd["A"].iloc[0] == pytest.approx(1.0)


class TestQuantiles:
    def test_monotone_factor_yields_monotone_spread(self) -> None:
        close = _random_close()
        fwd = forward_returns(close, horizon=21)
        # 스코어 = 미래수익 + 약한 노이즈 → 강한 단조성 기대
        scores = fwd + RNG.normal(0, fwd.std().mean() * 0.3, fwd.shape)
        report = analyze_quantiles(scores, fwd, n_quantiles=5)
        assert report.spread > 0
        assert report.monotonicity == pytest.approx(1.0)
        assert report.spread_t > 5


class TestOverfitting:
    def test_dsr_decreases_with_trials(self) -> None:
        good = pd.Series(RNG.normal(0.001, 0.01, 1000))
        dsr_1 = deflated_sharpe_ratio(good, n_trials=1)
        dsr_500 = deflated_sharpe_ratio(good, n_trials=500)
        assert 0.0 <= dsr_500 < dsr_1 <= 1.0

    def test_pbo_high_for_pure_noise(self) -> None:
        noise = pd.DataFrame(RNG.normal(0, 0.01, (240, 20)))
        result = probability_of_backtest_overfitting(noise, n_blocks=8)
        assert 0.25 <= result.pbo <= 0.85  # 노이즈 선택은 동전던지기 수준

    def test_pbo_low_for_dominant_strategy(self) -> None:
        m = pd.DataFrame(RNG.normal(0, 0.01, (240, 20)))
        m[0] = RNG.normal(0.01, 0.01, 240)  # 압도적 진짜 전략
        result = probability_of_backtest_overfitting(m, n_blocks=8)
        assert result.pbo < 0.2

    def test_pbo_rejects_insufficient_data(self) -> None:
        with pytest.raises(ValueError, match="필요"):
            probability_of_backtest_overfitting(pd.DataFrame(np.ones((10, 1))))


class TestCovariance:
    def test_shrinkage_intensity_bounded_and_psd(self) -> None:
        rets = pd.DataFrame(RNG.normal(0, 0.02, (60, 15)), columns=TICKERS[:15])
        cov, delta = ledoit_wolf_cc(rets)
        assert 0.0 <= delta <= 1.0
        assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-12

    def test_shrinks_harder_when_t_small(self) -> None:
        """
        타깃(상수상관)이 틀린 세계 — 이질적 베타의 팩터 구조 — 에서는
        관측이 늘수록 표본을 믿어야 하므로 δ 가 줄어야 한다.
        (iid 데이터에서는 타깃이 참값이라 δ→1 이 올바른 동작이다.)
        """

        def factor_returns(t: int) -> pd.DataFrame:
            betas = np.linspace(0.2, 2.0, 30)
            f = RNG.normal(0, 0.02, (t, 1))
            eps = RNG.normal(0, 0.01, (t, 30))
            return pd.DataFrame(f @ betas[None, :] + eps)

        _, d_wide = ledoit_wolf_cc(factor_returns(40))
        _, d_long = ledoit_wolf_cc(factor_returns(2000))
        assert d_wide > d_long  # 관측이 적을수록 타깃 쪽으로 강하게 수축


class TestWeights:
    @staticmethod
    def _rets(vols: list[float], n: int = 500) -> pd.DataFrame:
        data = {f"A{i}": RNG.normal(0, v, n) for i, v in enumerate(vols)}
        return pd.DataFrame(data)

    def test_risk_parity_equalizes_risk_contributions(self) -> None:
        rets = self._rets([0.01, 0.02, 0.04, 0.08])
        w = risk_parity(rets)
        cov = rets.cov().to_numpy()
        rc = w.to_numpy() * (cov @ w.to_numpy())
        assert rc.max() / rc.min() < 1.5  # 위험기여도 균등 (표본오차 허용)
        assert w.sum() == pytest.approx(1.0)
        # 고변동 자산일수록 낮은 비중
        assert w["A0"] > w["A3"]

    def test_hrp_valid_weights(self) -> None:
        base = RNG.normal(0, 0.02, (300, 1))
        block1 = base + RNG.normal(0, 0.005, (300, 3))  # 상관 블록
        block2 = RNG.normal(0, 0.02, (300, 3))  # 독립 블록
        rets = pd.DataFrame(np.hstack([block1, block2]), columns=[f"A{i}" for i in range(6)])
        w = hrp(rets)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
        # 상관 블록의 합산 비중 < 독립 블록 (분산효과 반영)
        assert w.iloc[:3].sum() < w.iloc[3:].sum()

    def test_mvo_respects_cap_and_tilts_to_score(self) -> None:
        rets = self._rets([0.02] * 5)
        scores = pd.Series([3.0, 0.0, 0.0, 0.0, -3.0], index=rets.columns)
        w = mvo(rets, scores, max_weight=0.35)
        assert w.sum() == pytest.approx(1.0)
        assert w.max() <= 0.35 + 1e-9
        assert w["A0"] == w.max()
        assert w["A4"] == w.min()

    def test_bl_confidence_controls_tilt(self) -> None:
        rets = self._rets([0.02] * 5)
        scores = pd.Series([3.0, 0.0, 0.0, 0.0, -3.0], index=rets.columns)
        w_low = black_litterman(rets, scores, view_confidence=0.01, max_weight=0.5)
        w_high = black_litterman(rets, scores, view_confidence=10.0, max_weight=0.5)
        # 확신이 높을수록 팩터 베팅이 커진다; 낮으면 시장(균등)으로 수렴
        assert w_high["A0"] > w_low["A0"]
        assert abs(w_low["A0"] - 0.2) < abs(w_high["A0"] - 0.2)

    def test_cap_and_normalize_waterfill(self) -> None:
        w = cap_and_normalize(pd.Series({"a": 0.9, "b": 0.05, "c": 0.05}), max_weight=0.4)
        assert w["a"] == pytest.approx(0.4)
        assert w.sum() == pytest.approx(1.0)
        assert w["b"] == pytest.approx(0.3)


class TestScore:
    def test_rank_normalize_is_centered(self) -> None:
        panel = pd.DataFrame(RNG.normal(50, 10, (5, 30)), columns=TICKERS[:30])
        z = rank_normalize(panel)
        assert abs(z.mean(axis=1)).max() < 0.05
        assert z.notna().all().all()

    def test_composite_masks_low_coverage(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        full = pd.DataFrame(RNG.normal(size=(3, 4)), index=idx, columns=list("abcd"))
        partial = full.copy()
        partial[["c", "d"]] = np.nan  # c,d 는 팩터 1개만 관측
        score = composite_score({"f1": full, "f2": partial}, min_coverage=0.75)
        assert score[["a", "b"]].notna().all().all()
        assert score[["c", "d"]].isna().all().all()


class TestSharpeConvention:
    """
    Sharpe 는 프로젝트 전체가 초과수익 기준으로 통일돼 있다.

    팩터 엔진이 rf 를 빼지 않으면 채택 관문(OOS Sharpe > 0.5)이 rf/변동성 만큼
    느슨해지고, 기존 VAA 지표와 숫자를 비교할 수 없다.
    """

    def test_factor_sharpe_subtracts_risk_free_rate(self) -> None:
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.0006, 0.012, 500))

        with_rf = annualized_sharpe(returns)
        without_rf = annualized_sharpe(returns, risk_free_rate=0.0)
        assert with_rf < without_rf, "무위험이자율이 차감되지 않았다"

    def test_factor_sharpe_default_matches_project_constant(self) -> None:
        """기본값이 config 단일 상수를 따라야 VAA 쪽과 규약이 같다."""
        rng = np.random.default_rng(11)
        returns = pd.Series(rng.normal(0.0006, 0.012, 500))

        assert annualized_sharpe(returns) == pytest.approx(
            annualized_sharpe(returns, risk_free_rate=RISK_FREE_RATE)
        )

    def test_legacy_metrics_default_matches_project_constant(self) -> None:
        """analysis/metrics.py 에 2% 가 따로 박혀 있던 문제의 회귀 방지."""
        rng = np.random.default_rng(13)
        returns = pd.Series(rng.normal(0.01, 0.03, 60))

        assert calculate_sharpe_ratio(returns) == pytest.approx(
            calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE)
        )

    def test_backtest_stats_sharpe_matches_walkforward(self) -> None:
        """`backtest` 와 `optimize` 가 같은 수익률에 같은 Sharpe 를 줘야 한다."""
        rng = np.random.default_rng(17)
        idx = pd.date_range("2021-01-01", periods=500, freq="B")
        returns = pd.Series(rng.normal(0.0006, 0.012, 500), index=idx)
        result = BacktestResult(
            returns=returns,
            equity=(1.0 + returns).cumprod(),
            holdings=pd.DataFrame(),
            turnover=pd.Series(dtype=float),
            exposure=pd.Series(dtype=float),
        )

        assert result.stats()["sharpe"] == pytest.approx(annualized_sharpe(returns))


class TestCalmarObjective:
    """
    Calmar = CAGR / |최대낙폭|. Sharpe 가 변동성으로 나누는 것과 달리
    **낙폭**으로 나누므로, 같은 변동성이라도 한 번에 깊게 빠지는 전략을
    더 강하게 벌한다. 실전에서 견딜 수 있는지를 보는 지표다.
    """

    def test_deeper_drawdown_scores_lower(self) -> None:
        from opt_portfolio.factor.optimize.walkforward import annualized_calmar

        idx = pd.date_range("2020-01-01", periods=756, freq="B")
        steady = pd.Series(0.0004, index=idx)
        crashed = steady.copy()
        crashed.iloc[100:130] = -0.02  # 한 달간 급락

        assert annualized_calmar(crashed) < annualized_calmar(steady)

    def test_no_drawdown_is_not_infinite(self) -> None:
        """낙폭이 0 이면 나눗셈이 폭발한다 — 탐색이 그 점을 붙잡으면 안 된다."""
        from opt_portfolio.factor.optimize.walkforward import annualized_calmar

        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        monotone = pd.Series(0.0005, index=idx)

        assert np.isfinite(annualized_calmar(monotone))

    def test_too_short_is_rejected(self) -> None:
        from opt_portfolio.factor.optimize.walkforward import annualized_calmar

        short = pd.Series(0.001, index=pd.date_range("2020-01-01", periods=30, freq="B"))

        assert annualized_calmar(short) == -np.inf


class TestDeflatedSharpeUnits:
    """
    DSR 의 sr_var 는 **기간(일별) 단위** SR 분산이다 (`deflated_sharpe_ratio`
    docstring: "일별이면 일별 그대로 — 연율화 금지").

    그런데 walk-forward 의 목적함수는 `annualized_sharpe` — 연율화된 값이다.
    그 분산을 그대로 넘기면 252배 부풀려지고, SR₀ 가 √252 ≈ 15.9배 커져
    DSR 이 실제와 무관하게 0 으로 짜부라진다. 이 저장소가 겪은 DAILY 시총
    10⁶배 버그와 같은 유형이다.
    """

    @staticmethod
    def _result(daily_sr_values: list[float]):
        from opt_portfolio.factor.optimize.search import SearchResult, Trial
        from opt_portfolio.factor.optimize.walkforward import Fold, WalkForwardResult

        rng = np.random.default_rng(3)
        returns = pd.Series(
            rng.normal(0.0008, 0.011, 2000),
            index=pd.date_range("2015-01-01", periods=2000, freq="B"),
        )
        ann = [v * np.sqrt(252) for v in daily_sr_values]  # 목적함수는 연율화 값을 낸다
        trials = [Trial(params={"n": i}, objective=v) for i, v in enumerate(ann)]
        fold = Fold(*pd.to_datetime(["2015-01-01", "2018-01-01", "2018-02-01", "2019-01-01"]))
        return WalkForwardResult(
            oos_returns=returns,
            folds=[fold],
            params_per_fold=[{"n": 0}],
            searches=[SearchResult(best_params={"n": 0}, best_objective=max(ann), trials=trials)],
        )

    def test_annualized_objectives_are_converted_to_period_units(self) -> None:
        """연율화 분산을 그대로 쓰면 DSR 이 0 으로 죽는다 — 변환돼야 한다."""
        from opt_portfolio.factor.research.overfitting import deflated_sharpe_ratio

        daily_sr = [0.02, 0.03, 0.04, 0.05]
        result = self._result(daily_sr)

        expected = deflated_sharpe_ratio(result.oos_returns, 4, float(np.var(daily_sr)))
        assert result.deflated_sharpe() == pytest.approx(expected, abs=1e-6)

    def test_not_crushed_to_zero_by_unit_error(self) -> None:
        result = self._result([0.02, 0.03, 0.04, 0.05])

        assert result.deflated_sharpe() > 0.5, "단위 오류로 DSR 이 0 에 붙었다"


class TestValueWeighting:
    """
    시총가중 — Hou·Xue·Zhang(2020)이 아노말리 검증의 표준으로 요구하는 방식.

    균등가중은 소형주에 사실상 레버리지를 거는 것과 같아, 호가 스프레드
    반동으로 수익이 부풀려진다. HXZ 는 이 하나를 바꾸는 것만으로 452개
    아노말리의 65%가 유의성을 잃는다고 보고한다. 균등가중에서만 살아남는
    결과는 신뢰할 수 없으므로 엔진이 이 검증을 지원해야 한다.
    """

    def test_weights_follow_market_cap(self) -> None:
        from opt_portfolio.factor.portfolio.weights import compute_weights

        rets = pd.DataFrame(RNG.normal(0, 0.02, (300, 3)), columns=["BIG", "MID", "SMALL"])
        caps = pd.Series({"BIG": 900e9, "MID": 100e9, "SMALL": 10e9})

        w = compute_weights(
            "value", rets, pd.Series(0.0, index=rets.columns), market_caps=caps, max_weight=1.0
        )

        assert w["BIG"] > w["MID"] > w["SMALL"]
        assert w.sum() == pytest.approx(1.0)

    def test_cap_limits_mega_cap_dominance(self) -> None:
        """상한이 없으면 초대형주 하나가 포트폴리오를 삼킨다."""
        from opt_portfolio.factor.portfolio.weights import compute_weights

        rets = pd.DataFrame(RNG.normal(0, 0.02, (300, 3)), columns=["BIG", "MID", "SMALL"])
        caps = pd.Series({"BIG": 9000e9, "MID": 100e9, "SMALL": 10e9})

        w = compute_weights(
            "value", rets, pd.Series(0.0, index=rets.columns), market_caps=caps, max_weight=0.5
        )

        assert w["BIG"] <= 0.5 + 1e-9

    def test_missing_caps_fall_back_to_equal(self) -> None:
        """시총이 없는 구간(초기 히스토리)에서 조용히 0 이 되면 안 된다."""
        from opt_portfolio.factor.portfolio.weights import compute_weights

        rets = pd.DataFrame(RNG.normal(0, 0.02, (300, 3)), columns=["A", "B", "C"])

        w = compute_weights(
            "value", rets, pd.Series(0.0, index=rets.columns), market_caps=None, max_weight=1.0
        )

        assert w.sum() == pytest.approx(1.0)
        assert w.std() == pytest.approx(0.0, abs=1e-9)
