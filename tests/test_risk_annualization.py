"""
연율화 주기 — 월별 수익률에 √252 를 쓰면 지표가 조용히 틀린다.

VAA 백테스트가 **월별** 수익률을 내는데 `RiskAnalyzer` 가 일별(252)로 연율화하고
있었다. 결과가 얼마나 어긋났는지 실측하면:

| 지표 | 보고값 | 실제 | 배수 |
|---|---|---|---|
| 연율 변동성 | 39.99% | 8.73% | ×4.58 (√252/√12) |
| Sharpe | 3.204 | 0.153 | ×20.9 (252/12 ÷ 4.58) |

CAGR·MDD 는 equity 곡선에서 나오므로 주기와 무관하고 정확했다. 그래서 **성과는
맞는데 위험지표만 틀린** 상태로 오래 남았다 — 두 숫자가 서로 모순인데도
(변동성 40% 인데 낙폭 17%) 아무도 보지 않았다.

`07-experiment-log.md` §2.1 의 DSR 단위 버그와 같은 계열이다. 그쪽은 연율화
Sharpe 의 분산을 기간 단위로 넘겨 판정을 뒤집었다. **연율화 주기는 이 저장소가
두 번 틀린 자리다.**
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.analysis.risk import RiskAnalyzer


def _monthly_returns(n: int = 180, mu: float = 0.005282, sd: float = 0.025189) -> pd.Series:
    """실측 VAA 월별 수익률과 같은 모수의 시계열."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2011-09-30", periods=n, freq="ME")
    return pd.Series(rng.normal(mu, sd, n), index=idx)


class TestPeriodsPerYear:
    """`periods_per_year` 를 받아 월별·일별을 구분해야 한다."""

    def test_monthly_volatility_uses_sqrt_12(self) -> None:
        r = _monthly_returns()
        vol = RiskAnalyzer().calculate_volatility(r, periods_per_year=12)

        assert vol == pytest.approx(r.std() * np.sqrt(12), rel=1e-9)

    def test_monthly_volatility_is_not_the_daily_number(self) -> None:
        """√252 를 쓰면 4.58배 부푼다 — 그 값이 나오면 실패다."""
        r = _monthly_returns()
        vol = RiskAnalyzer().calculate_volatility(r, periods_per_year=12)

        assert vol != pytest.approx(r.std() * np.sqrt(252), rel=1e-6)
        assert vol < 0.20, f"월별 8~9% 대여야 하는데 {vol:.1%} 다"

    def test_monthly_sharpe_matches_definition(self) -> None:
        r = _monthly_returns()
        rf = 0.05
        sharpe = RiskAnalyzer(risk_free_rate=rf).calculate_sharpe_ratio(r, periods_per_year=12)

        expected = (r.mean() * 12 - rf) / (r.std() * np.sqrt(12))
        assert sharpe == pytest.approx(expected, rel=1e-9)

    def test_monthly_sharpe_is_not_inflated_twentyfold(self) -> None:
        """분자를 252로 연율화하면 20.9배 부푼다 — 실측에서 3.20 이 나왔다."""
        r = _monthly_returns()
        sharpe = RiskAnalyzer(risk_free_rate=0.05).calculate_sharpe_ratio(r, periods_per_year=12)

        assert abs(sharpe) < 2.0, f"월별 Sharpe 가 {sharpe:.2f} — 연율화 주기를 의심하라"

    def test_daily_path_still_uses_252(self) -> None:
        """기본값은 일별이어야 기존 호출부가 안 깨진다."""
        rng = np.random.default_rng(1)
        idx = pd.date_range("2020-01-01", periods=504, freq="B")
        r = pd.Series(rng.normal(0.0004, 0.01, 504), index=idx)

        assert RiskAnalyzer().calculate_volatility(r) == pytest.approx(
            r.std() * np.sqrt(252), rel=1e-9
        )


class TestBacktestPassesMonthly:
    """백테스트는 월별 리밸런싱이므로 12를 넘겨야 한다."""

    def test_backtest_result_reports_monthly_metrics(self) -> None:
        from opt_portfolio.analysis.backtest import BacktestResult

        idx = pd.date_range("2011-09-30", periods=180, freq="ME")
        r = _monthly_returns()
        eq = pd.Series((1 + r).cumprod().values * 10000, index=idx)

        res = BacktestResult(
            strategy_name="test",
            equity_curve=eq,
            returns=r,
            transactions=[],
            initial_capital=10000.0,
            final_capital=float(eq.iloc[-1]),
        )
        res.calculate_metrics(years=15.0)

        assert res.volatility < 0.20, f"월별 변동성이 {res.volatility:.1%} 로 나온다"
        assert abs(res.sharpe_ratio) < 2.0, f"월별 Sharpe 가 {res.sharpe_ratio:.2f} 다"
