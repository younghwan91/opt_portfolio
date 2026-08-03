"""분기 그리드 전용 롤링 연산 — 일별 Rolling 노드와 그리드가 다르다."""

from __future__ import annotations

from dataclasses import dataclass

from opt_portfolio.factor.dsl.context import PanelContext
from opt_portfolio.factor.dsl.expr import Expr, GridError, Panel


@dataclass(frozen=True)
class QuarterlyRolling(Expr):
    """분기 그리드에서의 롤링 통계. 일별 그리드로 호출하면 즉시 에러."""

    child: Expr
    window: int
    how: str

    def eval(self, ctx: PanelContext) -> Panel:
        panel = self.child.eval(ctx)
        if panel.grid != "quarterly":
            raise GridError(
                f"quarterly_rolling_{self.how} 는 분기 그리드 전용입니다 "
                f"(받은 그리드: {panel.grid}). 표현식: {self.child.describe()}"
            )
        roll = panel.data.rolling(self.window, min_periods=max(4, self.window // 2))
        return panel.with_data(getattr(roll, self.how)())

    def describe(self) -> str:
        return f"q_{self.how}{self.window}({self.child.describe()})"


def quarterly_rolling_std(expr: Expr, window: int) -> Expr:
    return QuarterlyRolling(expr, window=window, how="std")


def quarterly_rolling_mean(expr: Expr, window: int) -> Expr:
    return QuarterlyRolling(expr, window=window, how="mean")
