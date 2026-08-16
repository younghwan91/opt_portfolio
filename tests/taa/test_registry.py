from __future__ import annotations

from opt_portfolio.taa.registry import MA_OVERLAY, REGISTERED, TRANCHE

EXPECTED = {
    "spy",
    "static_60_40",
    "vaa_g4",
    "baa_agg",
    "baa_bal",
    "baa_agg_ma",
    "baa_bal_ma",
    "baa_bal_tranche",
    "baa_bal_ma_tranche",
}


class TestRegistry:
    def test_exactly_nine_configurations(self) -> None:
        """DSR 의 n_trials 가 9 다. 늘리면 관문이 무의미해진다."""
        assert len(REGISTERED) == 9
        assert set(REGISTERED) == EXPECTED

    def test_vaa_uses_13612w_for_selection(self) -> None:
        assert REGISTERED["vaa_g4"].selection == "13612w"

    def test_baa_uses_sma13_for_selection(self) -> None:
        assert REGISTERED["baa_agg"].selection == "sma13"
        assert REGISTERED["baa_bal"].selection == "sma13"

    def test_vaa_canary_equals_its_offensive_universe(self) -> None:
        """VAA 의 병 — 경보기와 투자 대상이 같다."""
        spec = REGISTERED["vaa_g4"]
        assert set(spec.canary) == set(spec.offensive)

    def test_baa_canary_differs_from_offensive(self) -> None:
        """BAA 의 해법 — 분리한다."""
        spec = REGISTERED["baa_agg"]
        assert set(spec.canary) != set(spec.offensive)

    def test_baa_has_cash_dual_momentum_vaa_does_not(self) -> None:
        assert REGISTERED["baa_bal"].cash_ticker == "BIL"
        assert REGISTERED["vaa_g4"].cash_ticker is None

    def test_baa_balanced_holds_six_offensive_three_defensive(self) -> None:
        spec = REGISTERED["baa_bal"]
        assert spec.top_n_offensive == 6
        assert spec.top_n_defensive == 3

    def test_variant_flags_reference_registered_names(self) -> None:
        assert MA_OVERLAY <= set(REGISTERED)
        assert TRANCHE <= set(REGISTERED)

    def test_static_baseline_has_no_signals(self) -> None:
        assert REGISTERED["static_60_40"].static_weights == {"SPY": 0.6, "IEF": 0.4}
        assert REGISTERED["spy"].static_weights == {"SPY": 1.0}
