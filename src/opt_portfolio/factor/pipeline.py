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

import logging
from dataclasses import dataclass, field, replace

import pandas as pd

from opt_portfolio.factor.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from opt_portfolio.factor.backtest.timing import (
    momentum_exposure,
    volatility_target_exposure,
)
from opt_portfolio.factor.dsl.context import PanelContext
from opt_portfolio.factor.dsl.registry import REGISTRY, FactorSpec
from opt_portfolio.factor.optimize.search import Params
from opt_portfolio.factor.optimize.walkforward import Evaluator
from opt_portfolio.factor.portfolio.score import composite_score, rank_normalize
from opt_portfolio.factor.universe.filters import UniverseConfig, build_universe

logger = logging.getLogger(__name__)

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
    #: 변동성 타게팅 목표 (연율). None 이면 미사용.
    #: 이평 타이밍과 함께 켜면 두 익스포저를 곱한다 — 이진 차단 위에
    #: 연속 조절을 얹는 형태다.
    target_vol: float | None = None
    vol_window: int = 63
    vol_min_exposure: float = 0.0
    vol_max_exposure: float = 1.0
    benchmark: str = "SPY"
    signal_freq: str = "ME"  # 스코어 샘플링 그리드
    regime_conditional: bool = False  # 레짐별 팩터 가중 (research/regime.py)
    #: >0 이면 **학습 구간 안에서** 팩터 풀 중 상위 k개를 고른다.
    #: 사후에 사람이 고르면 그 선택은 DSR 시도 횟수에 안 들어가는
    #: 미정산 부채가 된다 — 선택을 학습 안으로 넣어 OOS 로 검증한다.
    select_top_k: int = 0
    #: 선별 기준. "ic" 는 개별 IC 상위 k개, "residual" 은 이미 고른 팩터에
    #: 회귀하고 남은 잔차의 IC 로 고르는 전진 선택이다. 전자는 겹치는 팩터가
    #: 자리를 다 차지한다 — 실측에서 성장 팩터가 19폴드 내내 0번 뽑혔다.
    select_method: str = "ic"
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
        self._panel_cache: dict[tuple, pd.DataFrame] = {}
        self._universe_cache: dict[UniverseConfig, pd.DataFrame] = {}
        self._regime_cache: dict[tuple, pd.DataFrame] = {}
        self._selection_cache: dict[tuple, list[str]] = {}

    # ------------------------------------------------------------ 팩터 패널
    def factor_panel(self, spec: FactorSpec, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        스코어링 표현식(방향·역수·중립화 처리 완료)의 신호일 그리드 패널.

        **신호일 그리드만 캐시한다.** 엔진은 리밸런싱 신호만 쓰는데 일별
        패널(종목 × 전 거래일)을 들고 있으면 메모리가 21배 커진다 —
        6,895종목 실적재에서 이 낭비가 OOM 을 두 번 냈다 (RSS 11.8GB).
        일별 원본은 계산 직후 버린다.
        """
        key = (spec.name, len(signal_dates), signal_dates[0], signal_dates[-1])
        if key not in self._panel_cache:
            daily = self.ctx.eval_daily(spec.scoring_expr())
            self._panel_cache[key] = daily.reindex(signal_dates, method="ffill")
            del daily
        return self._panel_cache[key]

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

    def selected_scores(
        self,
        config: StrategyConfig,
        train_end: pd.Timestamp,
        *,
        horizon: int = 21,
    ) -> pd.DataFrame:
        """
        `train_end` 까지의 정보로만 팩터를 고른 뒤 합성한 스코어.

        선택 결과는 (팩터 조합, train_end) 로 캐시한다 — 같은 폴드의 여러
        시도가 같은 선택을 공유하므로 재계산할 이유가 없다.
        """
        from opt_portfolio.factor.research.ic import forward_returns
        from opt_portfolio.factor.research.selection import SELECTORS

        key = (config.factors, config.select_top_k, config.select_method, pd.Timestamp(train_end))
        if key in self._selection_cache:
            names = self._selection_cache[key]
        else:
            select = SELECTORS[config.select_method]
            dates = self.signal_dates(config.signal_freq)
            panels = {s.name: self.factor_panel(s, dates) for s in config.resolved_factors()}
            fwd = forward_returns(self.close, horizon=horizon).reindex(dates)
            names = select(panels, fwd, end=pd.Timestamp(train_end), k=config.select_top_k)
            self._selection_cache[key] = names
            logger.info(
                "폴드 팩터 선택 (%s, ≤%s): %s",
                config.select_method,
                pd.Timestamp(train_end).date(),
                names,
            )

        dates = self.signal_dates(config.signal_freq)
        chosen = {
            s.name: self.factor_panel(s, dates)
            for s in config.resolved_factors()
            if s.name in names
        }
        return composite_score(chosen, None)

    def universe(self, config: UniverseConfig) -> pd.DataFrame:
        if config not in self._universe_cache:
            self._universe_cache[config] = build_universe(self.ctx, config)
        return self._universe_cache[config]

    def exposure(self, config: StrategyConfig) -> pd.Series | None:
        if config.timing_ma_days is None and config.target_vol is None:
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
        parts: list[pd.Series] = []
        if config.timing_ma_days is not None:
            parts.append(
                momentum_exposure(bench_close, config.timing_ma_days, config.timing_reentry_days)
            )
        if config.target_vol is not None:
            parts.append(
                volatility_target_exposure(
                    bench_close.pct_change(),
                    target_vol=config.target_vol,
                    window=config.vol_window,
                    min_exposure=config.vol_min_exposure,
                    max_exposure=config.vol_max_exposure,
                )
            )
        combined = parts[0]
        for extra in parts[1:]:
            combined = combined * extra.reindex(combined.index).fillna(1.0)
        return combined

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
            sectors=self.ctx.meta.get("sector"),
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
        # 수익률은 파라미터와 무관하다. 폴드 19개 × 시도 3개 = 백 번 넘게
        # 호출되는데 그때마다 다시 만들면 500MB 짜리 프레임을 백 번 할당하고
        # 버린다 — 해제된 아레나가 OS 로 돌아가지 않아 RSS 가 계단식으로 올랐다.
        daily_returns = self.close.pct_change(fill_method=None)

        def evaluate(
            params: Params,
            start: pd.Timestamp,
            end: pd.Timestamp,
            train_end: pd.Timestamp | None = None,
        ) -> pd.Series:
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
            if base.select_top_k > 0:
                # 학습 구간 끝 기준으로 고른다. search() 는 train 구간으로,
                # 최종 평가는 test 구간으로 호출되므로 두 경우 모두
                # **그 폴드의 학습 끝**을 넘겨야 한다 (아래 evaluator 참조).
                scores = self.selected_scores(base, train_end=train_end or start)
            elif base.regime_conditional:
                scores = self.regime_scores(base)
            else:
                scores = self.scores(base, factor_weights=weights)
            result = run_backtest(
                self.close,
                scores,
                bt_config,
                universe=universe_mask,
                market_caps=mcap,
                exposure=exposure,
                sectors=self.ctx.meta.get("sector"),
                start=start,
                end=end,
                returns=daily_returns,
            )
            return result.returns

        return evaluate
