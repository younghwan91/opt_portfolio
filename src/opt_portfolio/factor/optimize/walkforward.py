"""
Walk-Forward 파라미터 최적화 — 이 시스템에서 PO 의 유일한 공식 경로

프로토콜:
    ┌─ train (확장 윈도) ─┐ embargo ┌─ test ─┐
    │  탐색: 여기서만      │  h일    │ 평가만  │
    └─────────────────────┘         └────────┘
    폴드마다 train 에서 최적 파라미터를 고르고, 그 파라미터로 test 를
    한 번만 실행한다. 모든 폴드의 test 수익률을 이어붙인 것이
    전략의 유일한 공식 성과다.

퀀트 관점:
- embargo 는 순방향 수익률의 겹침 때문이다. train 마지막 신호의 보유기간이
  test 시작과 겹치면 train 이 test 정보를 흡수한다 (López de Prado 의
  purged CV 와 같은 논리). embargo ≥ 보유기간이어야 한다.
- 전체 표본 최적화 함수는 이 모듈에 **없다**. 원하면 search() 를 직접
  쓸 수 있지만, 그 결과에는 OOS 성과가 없으므로 보고할 수 없다.
- 최종 보고는 DSR 로 정산한다: n_trials = 전 폴드 시도 합계.
  같은 데이터에서 실험을 반복할수록 DSR 은 스스로 나빠진다 — 정직한 설계.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from opt_portfolio.factor.optimize.search import (
    Params,
    ParamSpace,
    SearchResult,
    search,
)
from opt_portfolio.factor.research.overfitting import deflated_sharpe_ratio

#: evaluate(params, start, end) → 해당 구간의 일별 수익률
Evaluator = Callable[[Params, pd.Timestamp, pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    calendar: pd.DatetimeIndex,
    *,
    min_train_years: float = 5.0,
    test_months: int = 12,
    embargo_days: int = 21,
) -> list[Fold]:
    """
    확장 윈도 분할.

    Args:
        embargo_days: train 종료와 test 시작 사이 간격 (거래일 아님, 달력일).
            보유기간(리밸런싱 주기) 이상으로 잡을 것 — 월간이면 21일 이상.
    """
    start, end = calendar[0], calendar[-1]
    first_test = start + pd.DateOffset(years=int(min_train_years))
    folds = []
    test_start = first_test
    while test_start < end:
        test_end = min(test_start + pd.DateOffset(months=test_months), end)
        train_end = test_start - pd.Timedelta(days=embargo_days)
        if train_end <= start or (test_end - test_start).days < 60:
            break
        folds.append(Fold(start, train_end, test_start, test_end))
        test_start = test_end
    if not folds:
        raise ValueError(
            f"표본이 부족합니다: {start.date()} ~ {end.date()} 에서 "
            f"train {min_train_years}년 + test {test_months}개월 분할 불가"
        )
    return folds


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series  # 이어붙인 OOS 수익률 — 공식 성과
    folds: list[Fold]
    params_per_fold: list[Params]
    searches: list[SearchResult] = field(repr=False, default_factory=list)

    @property
    def n_trials_total(self) -> int:
        return sum(s.n_trials for s in self.searches)

    def deflated_sharpe(self) -> float:
        """
        전 시도 횟수를 반영한 DSR.

        시도 간 SR 분산은 폴드별 탐색 로그에서 직접 추정한다 —
        보수적 기본값보다 실측이 낫다.
        """
        objectives = np.concatenate(
            [s.objectives()[np.isfinite(s.objectives())] for s in self.searches]
        )
        sr_var = float(np.var(objectives)) if len(objectives) > 1 else None
        return deflated_sharpe_ratio(self.oos_returns, max(self.n_trials_total, 1), sr_var)

    def sharpe(self, ann: int = 252) -> float:
        r = self.oos_returns.dropna()
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else np.nan

    def param_stability(self) -> pd.DataFrame:
        """
        폴드별 선택 파라미터 — 폴드마다 최적값이 널뛰면 그 파라미터는
        신호가 아니라 노이즈를 피팅하고 있다는 강한 경고다.
        """
        return pd.DataFrame(self.params_per_fold, index=range(len(self.folds)))


def annualized_sharpe(returns: pd.Series, ann: int = 252) -> float:
    """기본 목적함수. 관측 60일 미만이면 -inf (탐색에서 자연 도태)."""
    r = returns.dropna()
    if len(r) < 60 or r.std(ddof=1) == 0:
        return -np.inf
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann))


def run_walk_forward(
    evaluate: Evaluator,
    space: ParamSpace,
    calendar: pd.DatetimeIndex,
    *,
    objective: Callable[[pd.Series], float] = annualized_sharpe,
    method: str = "bayesian",
    n_trials_per_fold: int = 24,
    min_train_years: float = 5.0,
    test_months: int = 12,
    embargo_days: int = 21,
    seed: int = 0,
) -> WalkForwardResult:
    """
    Walk-forward PO 실행.

    Args:
        evaluate: (params, start, end) → 그 구간 일별 수익률.
            내부에서 백테스트를 돌리는 클로저 — 데이터 바인딩은 호출 측 책임.
        objective: train 수익률 → 스칼라 (클수록 좋음)

    Returns:
        WalkForwardResult. `.oos_returns` 가 공식 성과,
        `.deflated_sharpe()` 가 공식 유의성이다.
    """
    folds = walk_forward_folds(
        calendar,
        min_train_years=min_train_years,
        test_months=test_months,
        embargo_days=embargo_days,
    )

    oos_parts: list[pd.Series] = []
    chosen: list[Params] = []
    searches: list[SearchResult] = []

    for k, fold in enumerate(folds):
        result = search(
            lambda p: objective(evaluate(p, fold.train_start, fold.train_end)),
            space,
            method=method,
            n_trials=n_trials_per_fold,
            seed=seed + k,  # 폴드마다 다른 시드 — 같은 초기점 반복 방지
        )
        searches.append(result)
        chosen.append(result.best_params)
        oos_parts.append(evaluate(result.best_params, fold.test_start, fold.test_end))

    oos = pd.concat(oos_parts).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]
    return WalkForwardResult(
        oos_returns=oos, folds=folds, params_per_fold=chosen, searches=searches
    )
