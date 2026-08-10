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
