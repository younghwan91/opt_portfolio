from __future__ import annotations

import pandas as pd

from opt_portfolio.taa.strategy import StrategySpec, select_weights

D = pd.Timestamp("2020-06-30")


def _frames(mom: dict[str, float], sel: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(mom, index=[D]),
        pd.DataFrame(sel, index=[D]),
    )


SPEC = StrategySpec(
    name="test",
    canary=("SPY", "EEM"),
    offensive=("QQQ", "EEM"),
    defensive=("IEF", "BIL"),
    top_n_offensive=1,
    top_n_defensive=1,
    selection="sma13",
    cash_ticker="BIL",
)


class TestSelectWeights:
    def test_all_canary_positive_goes_offensive(self) -> None:
        mom, sel = _frames(
            {"SPY": 0.5, "EEM": 0.1, "QQQ": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.20, "EEM": 1.05, "IEF": 1.01, "BIL": 1.00},
        )
        w = select_weights(SPEC, mom, sel, D)

        assert w == {"QQQ": 1.0}

    def test_any_canary_negative_goes_defensive(self) -> None:
        """VAA 도 BAA 도 공통인 breadth 규칙."""
        mom, sel = _frames(
            {"SPY": 0.5, "EEM": -0.01, "QQQ": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.20, "EEM": 1.05, "IEF": 1.10, "BIL": 1.00},
        )
        w = select_weights(SPEC, mom, sel, D)

        assert w == {"IEF": 1.0}

    def test_defensive_asset_below_cash_is_replaced_by_cash(self) -> None:
        """BIL 을 못 이기면 현금 — SHY 를 연 0.05% 로 들고 있던 문제의 해법."""
        mom, sel = _frames(
            {"SPY": -0.1, "EEM": 0.1, "QQQ": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.20, "EEM": 1.05, "IEF": 0.95, "BIL": 1.00},
        )
        w = select_weights(SPEC, mom, sel, D)

        assert w == {"BIL": 1.0}

    def test_multiple_holdings_are_equal_weighted(self) -> None:
        spec = StrategySpec(
            name="t2",
            canary=("SPY",),
            offensive=("QQQ", "EEM", "IEF"),
            defensive=("BIL",),
            top_n_offensive=2,
            top_n_defensive=1,
            selection="sma13",
            cash_ticker="BIL",
        )
        mom, sel = _frames(
            {"SPY": 0.5, "QQQ": 0.0, "EEM": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.30, "EEM": 1.20, "IEF": 1.10, "BIL": 1.00},
        )
        w = select_weights(spec, mom, sel, D)

        assert w == {"QQQ": 0.5, "EEM": 0.5}

    def test_static_spec_ignores_signals(self) -> None:
        """60/40 기준선 — 아무 판단도 하지 않는다."""
        spec = StrategySpec(
            name="60/40",
            canary=(),
            offensive=(),
            defensive=(),
            top_n_offensive=0,
            top_n_defensive=0,
            selection="sma13",
            cash_ticker=None,
            static_weights={"SPY": 0.6, "IEF": 0.4},
        )
        mom, sel = _frames({"SPY": -9.0}, {"SPY": 0.1})

        assert select_weights(spec, mom, sel, D) == {"SPY": 0.6, "IEF": 0.4}

    def test_uses_13612w_when_selection_is_13612w(self) -> None:
        """VAA 는 선택도 13612W 로 한다 — BAA 와 갈리는 지점."""
        spec = StrategySpec(
            name="vaa",
            canary=("SPY",),
            offensive=("QQQ", "EEM"),
            defensive=("IEF",),
            top_n_offensive=1,
            top_n_defensive=1,
            selection="13612w",
            cash_ticker=None,
        )
        mom, sel = _frames(
            {"SPY": 0.5, "QQQ": 0.1, "EEM": 0.9, "IEF": 0.0},
            {"QQQ": 9.9, "EEM": 0.1, "IEF": 1.0},  # sma 는 QQQ 가 높지만
        )
        w = select_weights(spec, mom, sel, D)

        assert w == {"EEM": 1.0}  # 13612W 기준이라 EEM

    def test_cash_tie_boundary_replaces_defensive_asset(self) -> None:
        """Regression: cash-tie boundary untested.

        IEF 가 BIL 과 정확히 같으면 현금으로 바뀌어야 한다.
        If the comparison were `>=` instead of `>`, this test would pass
        but should fail — catching the boundary bug.
        """
        spec = StrategySpec(
            name="test_tie",
            canary=("SPY",),
            offensive=(),
            defensive=("IEF",),
            top_n_offensive=0,
            top_n_defensive=1,
            selection="sma13",
            cash_ticker="BIL",
        )
        mom, sel = _frames(
            {"SPY": -0.1, "IEF": 0.0, "BIL": 0.0},
            {"IEF": 1.00, "BIL": 1.00},  # IEF 와 BIL 이 정확히 같음
        )
        w = select_weights(spec, mom, sel, D)

        # IEF 가 cash 를 이기지 못하면 BIL 로 바뀌어야 함
        assert w == {"BIL": 1.0}

    def test_multi_pick_collapse_to_same_cash(self) -> None:
        """Regression: multi-pick collapse to same cash ticker untested.

        두 방어 자산이 모두 현금을 못 이기면 둘 다 현금으로 바뀌어서
        결과는 {BIL: 1.0} 이고 합은 정확히 1.0 이어야 한다.
        If the code used dict assignment instead of accumulation,
        the weights would be 0.5 instead of 1.0 — returns would halve silently.
        """
        spec = StrategySpec(
            name="test_multi_collapse",
            canary=("SPY",),
            offensive=(),
            defensive=("IEF", "TLT"),
            top_n_offensive=0,
            top_n_defensive=2,
            selection="sma13",
            cash_ticker="BIL",
        )
        mom, sel = _frames(
            {"SPY": -0.1, "IEF": 0.0, "TLT": 0.0, "BIL": 0.0},
            {"IEF": 0.95, "TLT": 0.98, "BIL": 1.00},  # 둘 다 BIL 을 못 이김
        )
        w = select_weights(spec, mom, sel, D)

        assert w == {"BIL": 1.0}
        assert sum(w.values()) == 1.0
