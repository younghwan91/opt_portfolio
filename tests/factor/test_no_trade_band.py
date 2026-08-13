"""
매매 유예구간(no-trade band) — 회전율을 줄여 비용을 아끼는 장치.

채택 전략의 회전율은 리밸런싱당 67%이고, 수수료 0.5% 기준 연 2.7%가
비용으로 나간다. 목적함수는 비용을 모르므로 이 손실은 최적화로 줄지 않는다.

표준 해법은 **진입과 청산에 다른 문턱을 두는 것**이다. 상위 N위 안에
들면 사고, N×배수 밖으로 밀려야 판다. 순위가 문턱 근처에서 진동하는
종목을 매 분기 사고파는 낭비가 사라진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.backtest.engine import select_with_band


def _scores(order: list[str]) -> pd.Series:
    """순위가 명시적인 스코어 — 앞쪽이 높다."""
    return pd.Series(np.arange(len(order), 0, -1, dtype=float), index=order)


class TestSelectWithBand:
    def test_no_band_is_plain_top_n(self) -> None:
        """배수 1.0 이면 기존 동작과 같아야 한다 — 기본값이 행동을 바꾸지 않는다."""
        row = _scores(["A", "B", "C", "D", "E"])

        got = select_with_band(row, held=["D", "E"], n=3, hold_multiple=1.0)

        assert list(got.index) == ["A", "B", "C"]

    def test_holds_incumbent_inside_band(self) -> None:
        """보유 종목은 N위 밖이어도 N×배수 안이면 유지한다."""
        row = _scores(["A", "B", "C", "D", "E", "F"])

        got = select_with_band(row, held=["D"], n=3, hold_multiple=2.0)

        assert "D" in got.index, "밴드 안 보유 종목을 팔았다"
        assert len(got) == 3

    def test_drops_incumbent_outside_band(self) -> None:
        row = _scores(["A", "B", "C", "D", "E", "F", "G"])

        got = select_with_band(row, held=["G"], n=3, hold_multiple=2.0)

        assert "G" not in got.index, "밴드 밖으로 밀린 종목을 계속 들고 있다"

    def test_incumbent_does_not_crowd_out_better_names(self) -> None:
        """유지되는 보유분을 뺀 나머지를 신규로 채운다 — 정원은 항상 N이다."""
        row = _scores(["A", "B", "C", "D", "E"])

        got = select_with_band(row, held=["D", "E"], n=3, hold_multiple=2.0)

        assert len(got) == 3
        assert "A" in got.index, "최상위 종목이 밀려났다"

    def test_empty_holdings_is_plain_top_n(self) -> None:
        row = _scores(["A", "B", "C", "D"])

        got = select_with_band(row, held=[], n=2, hold_multiple=3.0)

        assert list(got.index) == ["A", "B"]

    def test_held_name_no_longer_investable_is_dropped(self) -> None:
        """상장폐지 등으로 스코어가 사라진 보유 종목은 조용히 유지되면 안 된다."""
        row = _scores(["A", "B", "C"])

        got = select_with_band(row, held=["ZZZ"], n=2, hold_multiple=3.0)

        assert "ZZZ" not in got.index
        assert len(got) == 2


class TestTurnoverEffect:
    def test_band_reduces_turnover_when_ranks_oscillate(self) -> None:
        """
        이 기능의 존재 이유. 순위가 문턱 근처에서 진동할 때
        밴드가 없으면 매번 갈아타고, 있으면 유지한다.
        """
        # C 와 D 가 3~4위를 번갈아 차지한다
        rows = [_scores(["A", "B", "C", "D"]), _scores(["A", "B", "D", "C"])] * 6

        for multiple, expect in ((1.0, "갈아탐"), (2.0, "유지")):
            held: list[str] = []
            switches = 0
            for row in rows:
                picked = list(select_with_band(row, held=held, n=3, hold_multiple=multiple).index)
                if held:
                    switches += len(set(picked) - set(held))
                held = picked
            if expect == "유지":
                assert switches == 0, f"밴드가 있는데 {switches}번 갈아탔다"
            else:
                assert switches > 0, "밴드가 없는데 한 번도 안 갈아탔다"


class TestConfig:
    def test_default_preserves_current_behaviour(self) -> None:
        from opt_portfolio.factor.backtest.engine import BacktestConfig

        assert BacktestConfig().hold_multiple == 1.0

    def test_multiple_below_one_is_rejected(self) -> None:
        """청산 문턱이 진입보다 좁으면 밴드가 아니라 잡음이다."""
        from opt_portfolio.factor.backtest.engine import BacktestConfig

        with pytest.raises(ValueError, match="hold_multiple"):
            BacktestConfig(hold_multiple=0.5)
