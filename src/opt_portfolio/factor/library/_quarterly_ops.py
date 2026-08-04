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


def avg_balance(expr: Expr) -> Expr:
    """
    기초·기말 평균 = (전분기 + 당분기) / 2.

    벤더가 주는 `equityavg`/`assetsavg`/`invcapavg` 컬럼은 ARQ(as-reported
    분기) 차원에서 전부 null 이다 — 실데이터 적재에서 확인(2026-08-05).
    ROE·ROA·ROIC 의 분모는 정의상 기간 평균이므로 스톡 계열에서 직접
    계산한다. 벤더가 채워주든 말든 결과가 같고, 계산식이 코드에 드러난다.
    """
    return (expr + expr.lag(1)) / 2.0
