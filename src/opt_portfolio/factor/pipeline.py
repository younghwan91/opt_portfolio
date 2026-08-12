"""
전략 파이프라인 — 스토어부터 walk-forward PO 까지의 조립부

    PITStore.build_context()
        → FactorPipeline(ctx)
            → .run(config)                 # 단일 백테스트 (탐색 아님)
            → .evaluator(config)           # run_walk_forward 에 꽂는 클로저

퀀트 관점:
- 팩터 패널은 파라미터와 무관하므로 **한 번만** 계산해 캐시한다.
  PO 한 번에 수백 회 백테스트가 돌므로 여기가 성능의 전부다.
- 합성 스코어는 신호일(월말) 그리드로 샘플링한다 — 일별 전체 횡단면
  랭킹은 6,000종목 기준 수십 GB 연산인데, 엔진은 리밸런싱일 신호만 쓴다.
- evaluator 가 받는 파라미터 이름이 곧 PO 공간의 키다:
  n_stocks / rebalance / weighting / max_weight / cov_window /
  ir_scale / view_confidence / w_<카테고리> (카테고리 가중치)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from opt_portfolio.factor.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from opt_portfolio.factor.backtest.timing import momentum_exposure
from opt_portfolio.factor.dsl.context import PanelContext
from opt_portfolio.factor.dsl.registry import REGISTRY, FactorSpec
from opt_portfolio.factor.optimize.search import Params
from opt_portfolio.factor.optimize.walkforward import Evaluator
from opt_portfolio.factor.portfolio.score import composite_score, rank_normalize
from opt_portfolio.factor.universe.filters import UniverseConfig, build_universe

#: BacktestConfig 로 그대로 전달되는 PO 파라미터 키
_BT_KEYS = frozenset(
    {
        "n_stocks",
        "rebalance",
        "weighting",
        "max_weight",
        "cov_window",
        "ir_scale",
        "view_confidence",
    }
)


@dataclass(frozen=True)
class StrategyConfig:
    """전략 한 개의 완전한 선언 — JSON 직렬화 가능."""

    factors: tuple[str, ...]  # 사용할 팩터 이름
    factor_weights: dict[str, float] | None = None  # None = 균등
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    timing_ma_days: int | None = None  # None = 마켓타이밍 없음
    timing_reentry_days: int = 5
    benchmark: str = "SPY"
    signal_freq: str = "ME"  # 스코어 샘플링 그리드
    regime_conditional: bool = False  # 레짐별 팩터 가중 (research/regime.py)
    subscribed: tuple[str, ...] = ("SF1", "SEP")  # 구독 중인 데이터셋

    def resolved_factors(self) -> list[FactorSpec]:
        """구독 데이터셋으로 계산 가능한 팩터만 — 나머지는 경고 후 제외."""
        have = frozenset(self.subscribed)
        specs = []
        for name in self.factors:
            spec = REGISTRY.get(name)
            if spec.requires <= have:
                specs.append(spec)
        if not specs:
            raise ValueError(
                f"구독 데이터셋 {sorted(have)} 으로 계산 가능한 팩터가 없습니다: "
                f"{list(self.factors)}"
            )
        return specs


class FactorPipeline:
    """컨텍스트 1개 위에서 여러 전략/파라미터를 평가하는 실행기."""

    def __init__(self, ctx: PanelContext) -> None:
        self.ctx = ctx
        if "close" not in ctx.daily:
            raise ValueError("파이프라인에는 daily close 가 필요합니다")
        self.close = ctx.daily["close"]
        self._panel_cache: dict[str, pd.DataFrame] = {}
        self._universe_cache: dict[UniverseConfig, pd.DataFrame] = {}
        self._regime_cache: dict[tuple, pd.DataFrame] = {}

    # ------------------------------------------------------------ 팩터 패널
    def factor_panel(self, spec: FactorSpec, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """스코어링 표현식(방향·역수·중립화 처리 완료)의 신호일 그리드 패널."""
        if spec.name not in self._panel_cache:
            daily = self.ctx.eval_daily(spec.scoring_expr())
            self._panel_cache[spec.name] = daily
        return self._panel_cache[spec.name].reindex(signal_dates, method="ffill")

    def signal_dates(self, freq: str) -> pd.DatetimeIndex:
        cal = pd.DatetimeIndex(self.close.index)
        s = pd.Series(cal, index=cal)
        return pd.DatetimeIndex(s.resample(freq).last().dropna().to_numpy())

    def scores(
        self,
        config: StrategyConfig,
        factor_weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        dates = self.signal_dates(config.signal_freq)
        specs = config.resolved_factors()
        panels = {s.name: self.factor_panel(s, dates) for s in specs}
        weights = factor_weights or config.factor_weights
        return composite_score(panels, weights)

    def regime_scores(
        self,
        config: StrategyConfig,
        *,
        min_months: int = 60,
        horizon: int = 21,
    ) -> pd.DataFrame:
        """
        레짐 조건부 합성 스코어 — "지금과 비슷했던 때 통한 팩터"에 가중한다.

        각 신호일 t 의 가중치는 **t 시점까지 관측 가능한 IC** 로만 정한다.
        IC 는 순방향 수익을 쓰므로 t 에서 계산된 IC 는 t+horizon 이 지나야
        알 수 있다 — 그래서 한 달치를 더 잘라낸다. 이 두 겹의 지연이 없으면
        레짐 조건부 백테스트는 조용히 미래를 본다.

        관측이 `min_months` 에 못 미치거나 현재 레짐의 표본이 부족하면
        **균등 가중으로 후퇴한다.** 조건부 판단이 근거를 잃었을 때 돌아갈
        곳은 1/N 이다 (DeMiguel et al. 2009).
        """
        from opt_portfolio.factor.research.ic import forward_returns, rank_ic
        from opt_portfolio.factor.research.regime import classify

        # 레짐 가중은 탐색 파라미터(n_stocks 등)와 무관하다. 캐시하지 않으면
        # 폴드×시도마다 전 구간을 다시 계산해 walk-forward 가 사실상 멈춘다.
        key = (config.factors, config.benchmark, config.signal_freq, min_months, horizon)
        if key in self._regime_cache:
            return self._regime_cache[key]

        dates = self.signal_dates(config.signal_freq)
        specs = config.resolved_factors()
        panels = {s.name: self.factor_panel(s, dates) for s in specs}

        benchmark = self.close.get(config.benchmark)
        if benchmark is None:
            raise ValueError(f"레짐 판정에 벤치마크 '{config.benchmark}' 가격이 필요합니다")
        regimes = classify(benchmark).reindex(dates, method="ffill")

        fwd = forward_returns(self.close, horizon=horizon).reindex(dates)
        ic = pd.DataFrame({name: rank_ic(panel, fwd) for name, panel in panels.items()})

        normalized = {name: rank_normalize(panel) for name, panel in panels.items()}
        lag = max(1, horizon // 21) + 1  # 순방향 수익 확정 지연 + 여유 1개월
        out = pd.DataFrame(0.0, index=dates, columns=self.close.columns)

        for i, date in enumerate(dates):
            usable = ic.iloc[: max(0, i - lag)]
            regime_now = regimes.iloc[i] if i < len(regimes) else None
            weights: dict[str, float] = {}
            if len(usable) >= min_months and regime_now:
                same = usable[regimes.iloc[: len(usable)].to_numpy() == regime_now]
                if len(same) >= 12:
                    mean_ic = same.mean()
                    positive = mean_ic[mean_ic > 0]
                    if not positive.empty:
                        weights = (positive / positive.sum()).to_dict()
            if not weights:  # 근거 부족 → 1/N
                weights = dict.fromkeys(panels, 1.0 / len(panels))
            row = sum(normalized[n].loc[date] * w for n, w in weights.items())
            out.loc[date] = row

        self._regime_cache[key] = out
        return out

    def universe(self, config: UniverseConfig) -> pd.DataFrame:
        if config not in self._universe_cache:
            self._universe_cache[config] = build_universe(self.ctx, config)
        return self._universe_cache[config]

    def exposure(self, config: StrategyConfig) -> pd.Series | None:
        if config.timing_ma_days is None:
            return None
        bench = self.ctx.meta.get("benchmark_return")
        if bench is None:
            close_b = self.close.get(config.benchmark)
            if close_b is None:
                raise ValueError(
                    f"벤치마크 '{config.benchmark}' 가격이 없어 타이밍을 계산할 수 없습니다"
                )
            bench_close = close_b
        else:
            bench_close = (1.0 + bench.fillna(0.0)).cumprod()
        return momentum_exposure(bench_close, config.timing_ma_days, config.timing_reentry_days)

    # ------------------------------------------------------------ 실행
    def run(
        self,
        config: StrategyConfig,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> BacktestResult:
        """단일 백테스트. PO 가 아니라 확정 전략의 성과 확인용."""
        scores = self.regime_scores(config) if config.regime_conditional else self.scores(config)
        return run_backtest(
            self.close,
            scores,
            config.backtest,
            universe=self.universe(config.universe),
            market_caps=self.ctx.daily.get("mcap"),
            exposure=self.exposure(config),
            start=pd.Timestamp(start) if start else None,
            end=pd.Timestamp(end) if end else None,
        )

    def evaluator(self, base: StrategyConfig) -> Evaluator:
        """
        run_walk_forward 용 클로저.

        파라미터 해석:
        - _BT_KEYS 에 있는 키 → BacktestConfig 필드 교체
        - "w_<팩터명>" → 팩터 가중치 교체 (스코어 재합성 — 캐시된 패널 재사용)
        나머지 키는 에러 — 조용한 오타 무시는 PO 를 무의미하게 만든다.
        """
        universe_mask = self.universe(base.universe)
        exposure = self.exposure(base)
        mcap = self.ctx.daily.get("mcap")

        def evaluate(params: Params, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
            bt_kwargs = {k: v for k, v in params.items() if k in _BT_KEYS}
            weight_overrides = {k[2:]: float(v) for k, v in params.items() if k.startswith("w_")}
            unknown = set(params) - _BT_KEYS - {f"w_{n}" for n in weight_overrides}
            if unknown:
                raise KeyError(f"알 수 없는 PO 파라미터: {sorted(unknown)}")

            bt_config = replace(base.backtest, **bt_kwargs)
            weights = (
                {**(base.factor_weights or {}), **weight_overrides}
                if weight_overrides
                else base.factor_weights
            )
            scores = (
                self.regime_scores(base)
                if base.regime_conditional
                else self.scores(base, factor_weights=weights)
            )
            result = run_backtest(
                self.close,
                scores,
                bt_config,
                universe=universe_mask,
                market_caps=mcap,
                exposure=exposure,
                start=start,
                end=end,
            )
            return result.returns

        return evaluate
