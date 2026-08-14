"""
파라미터 앙상블 — 폴드마다 '최적값 하나'를 쓰는 데서 오는 불안정성 완화.

실측 문제: 채택 전 실행에서 `n_stocks` 가 폴드에 따라 15↔47 로 널뛰었다.
최적값이 이렇게 갈리면 그 파라미터는 시장 구조가 아니라 그 구간의 노이즈를
맞추고 있다는 뜻이다. 그런데 우리는 그 하나에 검증 구간 전체를 건다.

상위 k개 파라미터의 검증 수익률을 평균 내면 그 선택 오차가 상쇄된다.
목적함수가 거의 같은 여러 조합 중 하나를 고르는 것은 동전던지기이고,
동전을 여러 번 던져 평균 내는 편이 낫다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.optimize.search import SearchResult, Trial
from opt_portfolio.factor.optimize.walkforward import top_k_params


def _result(objectives: dict[int, float]) -> SearchResult:
    trials = [Trial(params={"n": n}, objective=v) for n, v in objectives.items()]
    best = max(trials, key=lambda t: t.objective)
    return SearchResult(best_params=best.params, best_objective=best.objective, trials=trials)


class TestTopKParams:
    def test_k_one_is_the_best_params(self) -> None:
        """k=1 이면 기존 동작과 같아야 한다 — 기본값이 행동을 바꾸지 않는다."""
        picked = top_k_params(_result({10: 0.5, 20: 0.9, 30: 0.7}), k=1)

        assert picked == [{"n": 20}]

    def test_returns_k_best_in_order(self) -> None:
        picked = top_k_params(_result({10: 0.5, 20: 0.9, 30: 0.7}), k=2)

        assert picked == [{"n": 20}, {"n": 30}]

    def test_k_larger_than_trials_is_safe(self) -> None:
        picked = top_k_params(_result({10: 0.5, 20: 0.9}), k=99)

        assert len(picked) == 2

    def test_non_finite_objectives_are_dropped(self) -> None:
        """-inf 는 탐색에서 도태된 조합이다 — 앙상블에 넣으면 안 된다."""
        picked = top_k_params(_result({10: -np.inf, 20: 0.9, 30: 0.4}), k=3)

        assert {"n": 10} not in picked
        assert len(picked) == 2

    def test_empty_search_returns_best_params(self) -> None:
        empty = SearchResult(best_params={"n": 7}, best_objective=0.0, trials=[])

        assert top_k_params(empty, k=3) == [{"n": 7}]


class TestEnsembleEffect:
    def test_averaging_reduces_dispersion(self) -> None:
        """
        이 기능의 존재 이유. 목적함수가 비슷한 조합들의 검증 성과가 제각각일 때,
        하나만 고르면 그 편차를 그대로 떠안고 평균은 그것을 줄인다.
        """
        rng = np.random.default_rng(4)
        idx = pd.date_range("2020-01-01", periods=252, freq="B")
        candidates = [pd.Series(rng.normal(0.0004, 0.012, 252), index=idx) for _ in range(5)]

        single_std = np.std([float(c.mean()) for c in candidates])
        ensemble = pd.concat(candidates, axis=1).mean(axis=1)

        assert ensemble.std() < max(c.std() for c in candidates)
        assert single_std > 0  # 개별 선택은 실제로 흩어져 있다

    def test_ensemble_preserves_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        parts = [pd.Series(0.001, index=idx), pd.Series(0.003, index=idx)]

        ensemble = pd.concat(parts, axis=1).mean(axis=1)

        assert ensemble.index.equals(idx)
        assert ensemble.iloc[0] == pytest.approx(0.002)
