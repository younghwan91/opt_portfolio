"""
팩터 표현식 트리 (Factor DSL)

150개 팩터를 150개 함수로 구현하지 않기 위한 선언적 표현식 레이어.
팩터를 `mcap / netinc` 처럼 선언하면, TTM/QoQ/YoY/가속 파생형은
트랜스폼 체이닝으로 자동 생성된다.

퀀트 관점:
- 재무 데이터는 분기 그리드, 가격 데이터는 일별 그리드에 산다.
  두 그리드를 섞는 순간 look-ahead 가 발생하므로 Panel 이 grid 를 들고 다니며
  혼합 연산 시 PIT 맵을 통해서만 분기→일별로 승격된다.
- .ttm()/.qoq()/.yoy() 는 분기 그리드에서만 정의된다 (일별에서 호출 시 즉시 에러).
  일별 forward-fill 된 값에 QoQ 를 걸면 조용히 0 이 나오는 버그가 흔하다.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from opt_portfolio.factor.dsl.context import PanelContext

Grid = Literal["quarterly", "daily"]

# 성장률 분모 가드: 분모 절대값이 이 비율(매출 대비) 미만이면 NaN.
# 0 근처 분모에서 성장률이 폭발하는 것을 막는다.
_GROWTH_DENOM_FLOOR = 1e-9


@dataclass(frozen=True)
class Panel:
    """평가 결과 — (date × ticker) 와이드 프레임 + 소속 그리드."""

    data: pd.DataFrame
    grid: Grid

    def with_data(self, data: pd.DataFrame) -> Panel:
        return Panel(data=data, grid=self.grid)


class Expr(ABC):
    """팩터 표현식 트리 노드."""

    # ------------------------------------------------------------------ 평가
    @abstractmethod
    def eval(self, ctx: PanelContext) -> Panel:
        """표현식을 평가해 Panel 을 반환한다."""

    @abstractmethod
    def describe(self) -> str:
        """사람이 읽는 표현식 문자열 (레지스트리 문서화용)."""

    # ------------------------------------------------------- 산술 연산자 오버로딩
    def __add__(self, other: Expr | float) -> Expr:
        return BinOp(self, _lift(other), operator.add, "+")

    def __radd__(self, other: Expr | float) -> Expr:
        return BinOp(_lift(other), self, operator.add, "+")

    def __sub__(self, other: Expr | float) -> Expr:
        return BinOp(self, _lift(other), operator.sub, "-")

    def __rsub__(self, other: Expr | float) -> Expr:
        return BinOp(_lift(other), self, operator.sub, "-")

    def __mul__(self, other: Expr | float) -> Expr:
        return BinOp(self, _lift(other), operator.mul, "*")

    def __rmul__(self, other: Expr | float) -> Expr:
        return BinOp(_lift(other), self, operator.mul, "*")

    def __truediv__(self, other: Expr | float) -> Expr:
        return BinOp(self, _lift(other), _safe_div, "/")

    def __rtruediv__(self, other: Expr | float) -> Expr:
        return BinOp(_lift(other), self, _safe_div, "/")

    def __neg__(self) -> Expr:
        return UnaryOp(self, lambda df: -df, "neg")

    def __abs__(self) -> Expr:
        return UnaryOp(self, lambda df: df.abs(), "abs")

    # -------------------------------------------------- 재무 트랜스폼 (분기 그리드)
    def ttm(self) -> Expr:
        """직전 4개 분기 합. 스톡 항목은 최신값을 그대로 쓴다."""
        return TTM(self)

    def qoq(self) -> Expr:
        """전분기 대비 증감률."""
        return Growth(self, periods=1, label="qoq")

    def yoy(self) -> Expr:
        """전년동기 대비 증감률."""
        return Growth(self, periods=4, label="yoy")

    def accel(self, periods: int = 4) -> Expr:
        """성장률의 차분 = 2차 미분. 성장률 팩터 위에만 의미가 있다."""
        return Diff(self, periods=periods, label="accel")

    def lag(self, periods: int) -> Expr:
        return Shift(self, periods)

    # ------------------------------------------------------ 가격 트랜스폼 (일별)
    def mom(self, days: int, skip: int = 0) -> Expr:
        """
        days 영업일 모멘텀. skip>0 이면 직전 skip 일을 제외한다.
        정통 12-1 모멘텀은 mom(252, skip=21).
        """
        return Momentum(self, days=days, skip=skip)

    def ma(self, days: int) -> Expr:
        return Rolling(self, days=days, how="mean")

    def rolling_std(self, days: int) -> Expr:
        return Rolling(self, days=days, how="std")

    def pct_change(self, days: int = 1) -> Expr:
        return PctChange(self, days=days)

    # ------------------------------------------------- 횡단면 트랜스폼 (그리드 무관)
    def winsor(self, p: float = 0.01) -> Expr:
        """횡단면 상하위 p% 클리핑."""
        return CrossSection(self, how="winsor", param=p)

    def zscore(self, by: str | None = None) -> Expr:
        """횡단면 표준화. by='sector' 면 섹터 내부에서 표준화."""
        return CrossSection(self, how="zscore", group=by)

    def rank(self, by: str | None = None) -> Expr:
        """횡단면 백분위 랭킹 (0~1)."""
        return CrossSection(self, how="rank", group=by)

    def neutralize(self, by: str | tuple[str, ...]) -> Expr:
        """
        섹터/사이즈 중립화. 지정 그룹에 대해 횡단면 회귀 후 잔차를 취한다.
        R&D 집약도처럼 섹터 편향이 심한 팩터에 필수.
        """
        keys = (by,) if isinstance(by, str) else tuple(by)
        return Neutralize(self, keys=keys)


# --------------------------------------------------------------------- 리프 노드


@dataclass(frozen=True)
class Field(Expr):
    """소스 데이터 필드 참조. 그리드는 스키마 메타에서 결정된다."""

    name: str

    def eval(self, ctx: PanelContext) -> Panel:
        return ctx.field(self.name)

    def describe(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Expr):
    value: float

    def eval(self, ctx: PanelContext) -> Panel:
        raise _ConstOnly(self.value)

    def describe(self) -> str:
        return repr(self.value)


class _ConstOnly(Exception):
    """상수는 단독 평가되지 않는다 — BinOp 가 흡수한다."""

    def __init__(self, value: float) -> None:
        super().__init__(f"bare constant {value}")
        self.value = value


def _lift(x: Expr | float) -> Expr:
    return x if isinstance(x, Expr) else Const(float(x))


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """0 나눗셈을 inf 가 아니라 NaN 으로 — inf 는 랭킹을 오염시킨다."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b.where(b.abs() > _GROWTH_DENOM_FLOOR)
    return out.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------- 연산자 노드


@dataclass(frozen=True)
class BinOp(Expr):
    left: Expr
    right: Expr
    fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
    symbol: str

    def eval(self, ctx: PanelContext) -> Panel:
        lhs, rhs = _eval_pair(self.left, self.right, ctx)
        aligned_l, aligned_r, grid = _align(lhs, rhs, ctx)
        return Panel(self.fn(aligned_l, aligned_r), grid)

    def describe(self) -> str:
        return f"({self.left.describe()} {self.symbol} {self.right.describe()})"


@dataclass(frozen=True)
class UnaryOp(Expr):
    child: Expr
    fn: Callable[[pd.DataFrame], pd.DataFrame]
    label: str

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        return panel.with_data(self.fn(panel.data))

    def describe(self) -> str:
        return f"{self.label}({self.child.describe()})"


def _eval_pair(left: Expr, right: Expr, ctx: PanelContext) -> tuple[Panel, Panel]:
    """한쪽이 상수면 다른 쪽 모양에 맞춰 브로드캐스트한다."""
    l_const = isinstance(left, Const)
    r_const = isinstance(right, Const)
    if l_const and r_const:
        raise ValueError("두 피연산자가 모두 상수인 표현식은 팩터가 아닙니다")
    if l_const:
        rhs = right.eval(ctx)
        return Panel(_broadcast(left.value, rhs.data), rhs.grid), rhs
    if r_const:
        lhs = left.eval(ctx)
        return lhs, Panel(_broadcast(right.value, lhs.data), lhs.grid)
    return left.eval(ctx), right.eval(ctx)


def _broadcast(value: float, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(value, index=like.index, columns=like.columns)


def _align(lhs: Panel, rhs: Panel, ctx: PanelContext) -> tuple[pd.DataFrame, pd.DataFrame, Grid]:
    """
    그리드가 다르면 분기 → 일별로 승격한다.
    승격은 반드시 PIT 맵(공시일 기준)을 통과하므로 미래 정보가 새지 않는다.
    """
    if lhs.grid == rhs.grid:
        left, right = lhs.data.align(rhs.data, join="outer")
        return left, right, lhs.grid

    quarterly, daily = (lhs, rhs) if lhs.grid == "quarterly" else (rhs, lhs)
    promoted = ctx.to_daily(quarterly.data)
    if lhs.grid == "quarterly":
        left, right = promoted.align(rhs.data, join="right")
        return left, right, "daily"
    left, right = lhs.data.align(promoted, join="left")
    return left, right, "daily"


# ------------------------------------------------------------- 재무 트랜스폼 노드


@dataclass(frozen=True)
class TTM(Expr):
    """
    직전 4개 분기 합.

    스톡 항목(자산·자본·차입금)은 합산이 의미 없으므로 최신값을 그대로 반환한다.
    이 구분을 스키마 메타(`kind`)에서 자동으로 읽어 처리한다 —
    총자산 TTM 을 4분기 합으로 계산하면 값이 4배가 되는 흔한 버그를 원천 차단.
    """

    child: Expr

    def eval(self, ctx: PanelContext) -> Panel:
        panel = _require_quarterly(self.child, ctx, "ttm()")
        if _is_stock(self.child, ctx):
            return panel
        return panel.with_data(panel.data.rolling(4, min_periods=4).sum())

    def describe(self) -> str:
        return f"ttm({self.child.describe()})"


@dataclass(frozen=True)
class Growth(Expr):
    """
    성장률 = (x_t - x_{t-n}) / |x_{t-n}|

    분모에 abs() 를 쓰는 이유: 적자 축소(-100 → -50)를 성장으로 잡기 위함.
    abs() 없이 계산하면 부호가 뒤집혀 개선을 악화로 읽는다.
    """

    child: Expr
    periods: int
    label: str

    def eval(self, ctx: PanelContext) -> Panel:
        panel = _require_quarterly(self.child, ctx, f"{self.label}()")
        cur = panel.data
        prev = cur.shift(self.periods)
        denom = prev.abs()
        return panel.with_data(_safe_div(cur - prev, denom))

    def describe(self) -> str:
        return f"{self.label}({self.child.describe()})"


@dataclass(frozen=True)
class Diff(Expr):
    """n기 차분. 성장률 위에 걸면 '가속'이 된다."""

    child: Expr
    periods: int
    label: str

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        return panel.with_data(panel.data - panel.data.shift(self.periods))

    def describe(self) -> str:
        return f"{self.label}({self.child.describe()})"


@dataclass(frozen=True)
class Shift(Expr):
    child: Expr
    periods: int

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        return panel.with_data(panel.data.shift(self.periods))

    def describe(self) -> str:
        return f"lag({self.child.describe()}, {self.periods})"


# --------------------------------------------------------------- 가격 트랜스폼 노드


@dataclass(frozen=True)
class Momentum(Expr):
    child: Expr
    days: int
    skip: int

    def eval(self, ctx: PanelContext) -> Panel:
        panel = _require_daily(self.child, ctx, "mom()")
        end = panel.data.shift(self.skip)
        start = panel.data.shift(self.days)
        return panel.with_data(_safe_div(end, start) - 1.0)

    def describe(self) -> str:
        tail = f", skip={self.skip}" if self.skip else ""
        return f"mom({self.child.describe()}, {self.days}{tail})"


@dataclass(frozen=True)
class Rolling(Expr):
    child: Expr
    days: int
    how: Literal["mean", "std", "sum", "min", "max"]

    def eval(self, ctx: PanelContext) -> Panel:
        panel = _require_daily(self.child, ctx, f"rolling {self.how}")
        roll = panel.data.rolling(self.days, min_periods=max(2, self.days // 2))
        return panel.with_data(getattr(roll, self.how)())

    def describe(self) -> str:
        return f"{self.how}{self.days}({self.child.describe()})"


@dataclass(frozen=True)
class PctChange(Expr):
    child: Expr
    days: int

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        return panel.with_data(panel.data.pct_change(self.days))

    def describe(self) -> str:
        return f"pct{self.days}({self.child.describe()})"


# ------------------------------------------------------------- 횡단면 트랜스폼 노드


@dataclass(frozen=True)
class CrossSection(Expr):
    child: Expr
    how: Literal["winsor", "zscore", "rank"]
    param: float = 0.01
    group: str | None = None

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        groups = ctx.groups(self.group, panel) if self.group else None
        return panel.with_data(_apply_cross_section(panel.data, self.how, self.param, groups))

    def describe(self) -> str:
        tail = f", by={self.group}" if self.group else ""
        return f"{self.how}({self.child.describe()}{tail})"


@dataclass(frozen=True)
class Neutralize(Expr):
    """
    횡단면 회귀 잔차. 그룹 더미(섹터) + 연속 통제변수(로그 시총)를 지원한다.

    R&D/매출 같은 팩터는 중립화 없이는 '기술주 섹터 베팅'과 구분되지 않는다.
    """

    child: Expr
    keys: tuple[str, ...]

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        design = ctx.design_matrix(self.keys, panel)
        return panel.with_data(_residualize(panel.data, design))

    def describe(self) -> str:
        return f"neutralize({self.child.describe()}, by={'+'.join(self.keys)})"


# ------------------------------------------------------------------ 내부 헬퍼


def _require_quarterly(child: Expr, ctx: PanelContext, op: str) -> Panel:
    panel = child.eval(ctx)
    if panel.grid != "quarterly":
        raise GridError(
            f"{op} 는 분기 그리드에서만 정의됩니다 (받은 그리드: {panel.grid}). "
            f"일별로 forward-fill 된 값에 걸면 성장률이 0 으로 뭉개집니다. "
            f"표현식: {child.describe()}"
        )
    return panel


def _require_daily(child: Expr, ctx: PanelContext, op: str) -> Panel:
    panel = child.eval(ctx)
    if panel.grid != "daily":
        raise GridError(
            f"{op} 는 일별 그리드에서만 정의됩니다 (받은 그리드: {panel.grid}). "
            f"표현식: {child.describe()}"
        )
    return panel


def _is_stock(expr: Expr, ctx: PanelContext) -> bool:
    """단일 필드 참조이고 그 필드가 스톡이면 True."""
    return isinstance(expr, Field) and ctx.field_kind(expr.name) == "stock"


def _apply_cross_section(
    data: pd.DataFrame,
    how: str,
    param: float,
    groups: pd.DataFrame | None,
) -> pd.DataFrame:
    if groups is None:
        return _cross_section_rows(data, how, param)

    out = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    for label, mask in _group_masks(groups):
        block = data.where(mask)
        out = out.combine_first(_cross_section_rows(block, how, param))
    return out.reindex(index=data.index, columns=data.columns)


def _cross_section_rows(data: pd.DataFrame, how: str, param: float) -> pd.DataFrame:
    if how == "rank":
        return data.rank(axis=1, pct=True)
    if how == "winsor":
        lo = data.quantile(param, axis=1)
        hi = data.quantile(1.0 - param, axis=1)
        return data.clip(lower=lo, upper=hi, axis=0)
    if how == "zscore":
        mu = data.mean(axis=1)
        sd = data.std(axis=1, ddof=0)
        return data.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    raise ValueError(f"알 수 없는 횡단면 연산: {how}")


def _group_masks(groups: pd.DataFrame):
    """그룹 라벨 프레임 → (라벨, 불리언 마스크) 시퀀스."""
    labels = pd.unique(groups.to_numpy().ravel())
    for label in labels:
        if pd.isna(label):
            continue
        yield label, groups.eq(label)


def _residualize(data: pd.DataFrame, design: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    날짜별 횡단면 OLS 잔차.

    design 은 {통제변수명: (date × ticker) 프레임}. 범주형은 상위에서
    이미 원-핫 프레임들로 펼쳐져 들어온다.
    """
    if not design:
        return data

    out = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    controls = list(design.values())

    for date in data.index:
        y = data.loc[date]
        cols = [c.loc[date] if date in c.index else pd.Series(np.nan, index=data.columns)
                for c in controls]
        frame = pd.concat([y.rename("__y__"), *[c.rename(f"x{i}") for i, c in enumerate(cols)]],
                          axis=1).dropna()
        if len(frame) < len(controls) + 2:
            continue
        x = np.column_stack([np.ones(len(frame)), frame.iloc[:, 1:].to_numpy(dtype=float)])
        y_vec = frame["__y__"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y_vec, rcond=None)
        out.loc[date, frame.index] = y_vec - x @ beta
    return out


class GridError(ValueError):
    """분기/일별 그리드를 잘못 섞었을 때. look-ahead 를 사전 차단하는 가드."""


# 팩터 정의에서 쓰는 축약 네임스페이스: `F.netinc`, `F.mcap` 형태
class _FieldNamespace:
    """`F.revenue` 처럼 점 접근으로 Field 를 만드는 편의 객체."""

    def __getattr__(self, name: str) -> Field:
        if name.startswith("_"):
            raise AttributeError(name)
        return Field(name)

    def __getitem__(self, name: str) -> Field:
        return Field(name)


F = _FieldNamespace()
