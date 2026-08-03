"""
가격 팩터 (27개)

퀀트 관점:
- 1개월 모멘텀은 **반전(reversal)** 이 지배한다 → direction=-1.
  나머지 기간은 모멘텀 방향(+1). 부호를 통일해버리면 두 효과가 상쇄된다.
- 정통 모멘텀은 12-1(직전 1개월 제외)이다. 그냥 12개월을 쓰면 최근 1개월의
  반전 효과가 12개월 모멘텀 신호를 갉아먹는다. MOM_12_1 을 추가 등록한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from opt_portfolio.factor.dsl.context import PanelContext
from opt_portfolio.factor.dsl.expr import Expr, F, Panel, _require_daily
from opt_portfolio.factor.dsl.registry import factor

_SEP = ("SEP",)
_RET = F.close.pct_change(1)
_ANN = 252 ** 0.5

# ------------------------------------------------------------------ 모멘텀

_MOMENTUM_WINDOWS = [(1, 21), (3, 63), (6, 126), (12, 252)]

MOMENTUM_FACTORS = {
    months: factor(
        f"MOM_{months}M",
        F.close.mom(days),
        category="price",
        label=f"{months}개월 모멘텀",
        direction=-1 if months == 1 else 1,
        requires=_SEP,
        notes="단기 반전 — 낮을수록 좋음" if months == 1 else "",
    )
    for months, days in _MOMENTUM_WINDOWS
}

MOM_12_1 = factor(
    "MOM_12_1",
    F.close.mom(252, skip=21),
    category="price",
    label="12-1개월 모멘텀",
    requires=_SEP,
    notes="정통 Jegadeesh-Titman 모멘텀. 직전 1개월 반전 효과를 제거",
)

CLOSE_UNADJ = factor(
    "CLOSE_UNADJ", F.closeunadj, category="price", label="종가 (수정 전)",
    requires=_SEP, notes="팩터가 아니라 페니스톡 필터용",
)

# ------------------------------------------------------- 위험조정 수익 (샤프/소르티노)


@dataclass(frozen=True)
class DownsideStd(Expr):
    """하방 표준편차 — Sortino 분모. 음수 수익률만으로 계산한다."""

    child: Expr
    days: int

    def eval(self, ctx: PanelContext) -> Panel:
        panel = _require_daily(self.child, ctx, "downside_std")
        downside = panel.data.where(panel.data < 0)
        roll = downside.rolling(self.days, min_periods=max(2, self.days // 4))
        return panel.with_data(roll.std())

    def describe(self) -> str:
        return f"downside_std{self.days}({self.child.describe()})"


def sharpe_expr(days: int) -> Expr:
    return _RET.ma(days) / _RET.rolling_std(days) * _ANN


def sortino_expr(days: int) -> Expr:
    return _RET.ma(days) / DownsideStd(_RET, days) * _ANN


SHARPE = factor("SHARPE", sharpe_expr(252), category="price", label="샤프비율",
                requires=_SEP)
SORTINO = factor("SORTINO", sortino_expr(252), category="price", label="Sortino 비율",
                 requires=_SEP)

_RISK_ADJ_WINDOWS = [20, 60, 120, 200]

#: 샤프/소르티노 '모멘텀' = 해당 기간 위험조정수익의 변화량
SHARPE_MOM_FACTORS = {
    d: factor(
        f"SHARPE_MOM_{d}D",
        sharpe_expr(d) - sharpe_expr(d).lag(d),
        category="price",
        label=f"샤프비율 모멘텀 ({d}일)",
        requires=_SEP,
    )
    for d in _RISK_ADJ_WINDOWS
}

SORTINO_MOM_FACTORS = {
    d: factor(
        f"SORTINO_MOM_{d}D",
        sortino_expr(d) - sortino_expr(d).lag(d),
        category="price",
        label=f"Sortino 비율 모멘텀 ({d}일)",
        requires=_SEP,
    )
    for d in _RISK_ADJ_WINDOWS
}

# ------------------------------------------------------------------ RSI


@dataclass(frozen=True)
class RSI(Expr):
    """Wilder RSI. 지수가중 평활(alpha = 1/period)을 쓴다."""

    child: Expr
    period: int

    def eval(self, ctx: PanelContext) -> Panel:
        panel = _require_daily(self.child, ctx, "RSI")
        delta = panel.data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        alpha = 1.0 / self.period
        avg_gain = gain.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()
        rs = avg_gain / avg_loss.where(avg_loss > 0)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return panel.with_data(rsi.replace([np.inf, -np.inf], 100.0))

    def describe(self) -> str:
        return f"rsi{self.period}({self.child.describe()})"


RSI_FACTORS = {
    p: factor(
        f"RSI_{p}D", RSI(F.close, p), category="price", label=f"RSI ({p}일)",
        direction=-1, requires=_SEP,
        notes="과매수 회피 관점에서 낮을수록 좋음 — 방향은 IC 검증으로 확정할 것",
    )
    for p in (9, 14, 25)
}

# ------------------------------------------------------------------ 베타


@dataclass(frozen=True)
class Beta(Expr):
    """벤치마크 대비 롤링 베타. 벤치마크 수익률은 컨텍스트가 제공한다."""

    child: Expr
    days: int
    benchmark: str = "benchmark_return"

    def eval(self, ctx: PanelContext) -> Panel:
        panel = _require_daily(self.child, ctx, "beta")
        mkt = _benchmark_series(ctx, self.benchmark, panel.data.index)
        min_p = max(20, self.days // 2)
        cov = panel.data.rolling(self.days, min_periods=min_p).cov(mkt)
        var = mkt.rolling(self.days, min_periods=min_p).var()
        return panel.with_data(cov.div(var.where(var > 0), axis=0))

    def describe(self) -> str:
        return f"beta{self.days}({self.child.describe()})"


def _benchmark_series(ctx: PanelContext, name: str, index: pd.Index) -> pd.Series:
    series = ctx.meta.get(name)
    if series is None:
        raise KeyError(
            f"벤치마크 수익률 '{name}' 이(가) 컨텍스트에 없습니다. "
            f"PanelContext.meta['{name}'] 에 일별 시장수익률 시리즈를 넣으세요."
        )
    return series.reindex(index)


BETA_252 = factor("BETA", Beta(_RET, 252), category="price", label="베타",
                  direction=-1, requires=_SEP, notes="저베타 이상현상")
BETA_60 = factor("BETA_60D", Beta(_RET, 60), category="price", label="베타 (60일)",
                 direction=-1, requires=_SEP)
ABS_BETA_252 = factor("ABS_BETA", abs(Beta(_RET, 252)), category="price",
                      label="절대값 베타", direction=-1, requires=_SEP,
                      notes="0 에 가까울수록 시장중립")
ABS_BETA_60 = factor("ABS_BETA_60D", abs(Beta(_RET, 60)), category="price",
                     label="절대값 베타 (60일)", direction=-1, requires=_SEP)

# ------------------------------------------------------------------ 거래대금 회전율

TURNOVER = factor(
    "TURNOVER",
    (F.volume * F.close).ma(20) / F.mcap,
    category="price",
    label="거래대금 회전율",
    direction=-1,
    requires=_SEP,
    notes="저회전율 프리미엄(유동성 프리미엄). 유동성 필터와 역할이 다름 — "
          "필터는 체결 가능성, 이 팩터는 초과수익 원천",
)
