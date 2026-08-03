"""백테스트 엔진 (체결 규약, 비용, 타이밍) + walk-forward PO 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.backtest.costs import CostModel
from opt_portfolio.factor.backtest.engine import BacktestConfig, run_backtest
from opt_portfolio.factor.backtest.timing import combine_exposures, momentum_exposure
from opt_portfolio.factor.optimize.search import grid_params, search
from opt_portfolio.factor.optimize.walkforward import (
    run_walk_forward,
    walk_forward_folds,
)

RNG = np.random.default_rng(11)
DATES = pd.date_range("2020-01-01", periods=500, freq="B")


def _flat_scores(close: pd.DataFrame, favorite: str) -> pd.DataFrame:
    scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    scores[favorite] = 1.0
    return scores


class TestEngine:
    def test_single_asset_tracks_buy_and_hold(self) -> None:
        rets = RNG.normal(0.0005, 0.015, len(DATES))
        close = pd.DataFrame(
            {"A": 100 * np.exp(np.cumsum(rets)), "B": np.full(len(DATES), 50.0)},
            index=DATES,
        )
        result = run_backtest(
            close,
            _flat_scores(close, "A"),
            BacktestConfig(n_stocks=1, rebalance="ME", cost=CostModel.zero()),
        )
        # 첫 체결 이후 구간에서 포트폴리오 수익률 == 자산 수익률
        active = result.returns[result.returns != 0.0]
        asset = close["A"].pct_change().reindex(active.index)
        pd.testing.assert_series_equal(
            active, asset, check_names=False, atol=1e-12, rtol=0.0
        )

    def test_no_lookahead_signal_jump_not_captured(self) -> None:
        """신호일에 이미 발생한 점프 수익을 포트폴리오가 얻으면 안 된다."""
        jump_day = 250
        a = np.full(len(DATES), 100.0)
        a[jump_day:] = 150.0  # jump_day 에 +50%
        close = pd.DataFrame({"A": a, "B": np.full(len(DATES), 50.0)}, index=DATES)

        # 점프 당일부터 A 를 최상위로 미는 스코어 (신호가 점프를 '알고' 있음)
        scores = pd.DataFrame(0.0, index=DATES, columns=["A", "B"])
        scores.loc[DATES[jump_day]:, "A"] = 10.0
        scores["B"] = 1.0

        result = run_backtest(
            close,
            scores,
            BacktestConfig(n_stocks=1, rebalance="W-FRI", cost=CostModel.zero()),
        )
        # A 는 점프 이후 평평하므로, t+1 체결이면 점프는 절대 못 잡는다
        assert result.equity.iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_costs_reduce_equity(self) -> None:
        close = pd.DataFrame(
            100 * np.exp(np.cumsum(RNG.normal(0, 0.02, (len(DATES), 10)), axis=0)),
            index=DATES,
            columns=[f"T{i}" for i in range(10)],
        )
        scores = pd.DataFrame(
            RNG.normal(size=close.shape), index=DATES, columns=close.columns
        )
        base = BacktestConfig(n_stocks=3, rebalance="ME", cost=CostModel.zero())
        costly = BacktestConfig(
            n_stocks=3, rebalance="ME", cost=CostModel(commission_bps=50, slippage_bps=50)
        )
        eq_free = run_backtest(close, scores, base).equity.iloc[-1]
        eq_cost = run_backtest(close, scores, costly).equity.iloc[-1]
        assert eq_cost < eq_free

    def test_zero_exposure_flattens_portfolio(self) -> None:
        close = pd.DataFrame(
            {"A": 100 * np.exp(np.cumsum(RNG.normal(0, 0.02, len(DATES))))},
            index=DATES,
        )
        result = run_backtest(
            close,
            _flat_scores(close, "A"),
            BacktestConfig(n_stocks=1, rebalance="ME", cost=CostModel.zero()),
            exposure=pd.Series(0.0, index=DATES),
        )
        assert result.equity.iloc[-1] == pytest.approx(1.0)

    def test_delisting_moves_weight_to_cash(self) -> None:
        """세그먼트 도중 상장폐지 → 이후 수익률 0, NaN 전파 없음."""
        a = np.full(len(DATES), 100.0, dtype=float)
        a[300:] = np.nan  # 300일째 상장폐지
        close = pd.DataFrame({"A": a, "B": np.full(len(DATES), 50.0)}, index=DATES)
        result = run_backtest(
            close,
            _flat_scores(close, "A"),
            BacktestConfig(n_stocks=1, rebalance="QE", cost=CostModel.zero()),
        )
        assert result.returns.notna().all()
        assert np.isfinite(result.equity.iloc[-1])


class TestTiming:
    def test_exposure_zero_below_ma(self) -> None:
        price = pd.Series(
            np.linspace(100, 50, 300), index=pd.date_range("2020-01-01", periods=300)
        )
        exp = momentum_exposure(price, ma_days=50)
        assert (exp.iloc[60:] == 0.0).all()  # 하락 추세 → 이평 하회 → 현금

    def test_reentry_hysteresis_delays_reentry(self) -> None:
        # V자 반등: 하락 후 상승
        down = np.linspace(100, 60, 150)
        up = np.linspace(60, 120, 150)
        price = pd.Series(
            np.concatenate([down, up]),
            index=pd.date_range("2020-01-01", periods=300),
        )
        immediate = momentum_exposure(price, ma_days=50, reentry_days=0)
        delayed = momentum_exposure(price, ma_days=50, reentry_days=10)
        assert immediate.sum() > delayed.sum()  # 히스테리시스 = 늦은 복귀
        # 복귀 시점 차이 == reentry_days
        first_imm = (immediate.iloc[150:] == 1.0).idxmax()
        first_del = (delayed.iloc[150:] == 1.0).idxmax()
        assert (first_del - first_imm).days >= 9

    def test_combined_exposure_is_product(self) -> None:
        idx = pd.date_range("2020-01-01", periods=5)
        a = pd.Series([1, 1, 0.5, 1, 1], index=idx, dtype=float)
        b = pd.Series([1, 0, 1, 0.5, 1], index=idx, dtype=float)
        combined = combine_exposures(a, b)
        assert combined.iloc[2] == pytest.approx(0.5)
        assert combined.iloc[1] == pytest.approx(0.0)
        assert combined.iloc[3] == pytest.approx(0.5)


class TestSearch:
    SPACE = {
        "x": ("float", 0.0, 1.0),
        "mode": ("cat", ["fast", "slow"]),
        "n": ("int", 1, 10),
    }

    @staticmethod
    def _objective(p: dict) -> float:
        return -((p["x"] - 0.7) ** 2) + (0.1 if p["mode"] == "fast" else 0.0)

    def test_grid_enumerates_full_lattice(self) -> None:
        combos = grid_params(self.SPACE, steps=3)
        assert len(combos) == 3 * 2 * 3

    def test_bayesian_concentrates_near_optimum(self) -> None:
        result = search(
            self._objective, self.SPACE, method="bayesian", n_trials=40, seed=3
        )
        assert abs(result.best_params["x"] - 0.7) < 0.15
        assert result.best_params["mode"] == "fast"
        assert result.n_trials == 40  # 전 시도가 기록되었는가

    def test_failed_trials_recorded_as_neg_inf(self) -> None:
        def flaky(p: dict) -> float:
            if p["x"] < 0.5:
                raise RuntimeError("boom")
            return p["x"]

        result = search(flaky, self.SPACE, method="random", n_trials=20, seed=1)
        objs = result.objectives()
        assert np.isneginf(objs).any()      # 실패도 기록
        assert np.isfinite(result.best_objective)


class TestWalkForward:
    def test_folds_never_overlap_and_respect_embargo(self) -> None:
        cal = pd.date_range("2010-01-01", "2020-12-31", freq="B")
        folds = walk_forward_folds(cal, min_train_years=3, embargo_days=21)
        assert len(folds) >= 5
        for f in folds:
            assert (f.test_start - f.train_end).days >= 21
        for prev, nxt in zip(folds, folds[1:]):
            assert nxt.test_start >= prev.test_end

    def test_walk_forward_recovers_true_parameter(self) -> None:
        """진짜 최적값이 x=0.6 인 합성 세계에서 폴드별 선택이 그 근방인가."""
        cal = pd.date_range("2010-01-01", "2020-12-31", freq="B")

        def evaluate(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
            days = pd.date_range(start, end, freq="B")
            seed = int(start.value % 2**31)
            noise = np.random.default_rng(seed).normal(0, 0.01, len(days))
            mu = 0.0008 * (1.0 - 8.0 * (params["x"] - 0.6) ** 2)
            return pd.Series(mu + noise, index=days)

        result = run_walk_forward(
            evaluate,
            {"x": ("float", 0.0, 1.0)},
            cal,
            method="bayesian",
            n_trials_per_fold=16,
            min_train_years=4,
            seed=5,
        )
        stability = result.param_stability()
        assert (stability["x"] - 0.6).abs().median() < 0.2
        assert len(result.oos_returns) > 1000
        assert 0.0 <= result.deflated_sharpe() <= 1.0
        assert result.n_trials_total == 16 * len(result.folds)
