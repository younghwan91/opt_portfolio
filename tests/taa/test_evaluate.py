from __future__ import annotations

import numpy as np
import pandas as pd

from opt_portfolio.taa.evaluate import ADOPTION, common_window, evaluate_all, summarize, verdict


def _returns(mu: float, sd: float, n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2007-06-30", periods=n, freq="ME")
    return pd.Series(rng.normal(mu, sd, n), index=idx)


class TestSummarize:
    def test_reports_cagr_mdd_calmar(self) -> None:
        row = summarize("x", _returns(0.008, 0.03))

        assert {"cagr", "mdd", "calmar", "sharpe", "vol"} <= set(row)
        assert row["calmar"] == row["cagr"] / abs(row["mdd"])

    def test_annualizes_monthly_with_twelve(self) -> None:
        """월별인데 252 로 연율화하면 이 저장소가 세 번째로 같은 실수를 한다."""
        r = _returns(0.008, 0.03)
        row = summarize("x", r)

        assert row["vol"] == float(r.std() * np.sqrt(12))


class TestVerdict:
    def test_rejects_when_drawdown_exceeds_limit(self) -> None:
        m = pd.DataFrame([{"name": "a", "mdd": -0.35, "calmar": 2.0, "dsr": 0.99}]).set_index(
            "name"
        )
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "MDD" in out.loc["a", "reason"]

    def test_rejects_when_dsr_below_gate(self) -> None:
        m = pd.DataFrame([{"name": "a", "mdd": -0.15, "calmar": 2.0, "dsr": 0.80}]).set_index(
            "name"
        )
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "DSR" in out.loc["a", "reason"]

    def test_rejects_when_calmar_below_baseline(self) -> None:
        m = pd.DataFrame([{"name": "a", "mdd": -0.15, "calmar": 0.4, "dsr": 0.99}]).set_index(
            "name"
        )
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "60/40" in out.loc["a", "reason"]

    def test_adopts_only_when_all_gates_pass(self) -> None:
        m = pd.DataFrame([{"name": "a", "mdd": -0.15, "calmar": 0.9, "dsr": 0.99}]).set_index(
            "name"
        )
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert out.loc["a", "adopted"]

    def test_high_pbo_rejects_everything(self) -> None:
        """PBO 가 주 관문이다 — 넘으면 개별 성적과 무관하게 전부 기각."""
        m = pd.DataFrame([{"name": "a", "mdd": -0.10, "calmar": 3.0, "dsr": 1.0}]).set_index("name")
        out = verdict(m, pbo=0.6, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "PBO" in out.loc["a", "reason"]

    def test_adoption_constants_match_the_spec(self) -> None:
        assert ADOPTION["mdd_limit"] == -0.20
        assert ADOPTION["dsr_gate"] == 0.95
        assert ADOPTION["pbo_limit"] == 0.5

    def test_baseline_row_is_not_rejected_against_its_own_calmar(self) -> None:
        """`baseline_calmar` 는 static_60_40 자신의 Calmar 에서 뽑는다. 그
        행에도 `calmar <= baseline_calmar` 를 그대로 적용하면 자기 자신과
        비교해 "초과"가 성립할 수 없어 다른 성적과 무관하게 항상 기각된다
        — `baseline_name` 으로 그 행만 이 관문에서 뺀다.
        """
        m = pd.DataFrame(
            [
                {"name": "static_60_40", "mdd": -0.15, "calmar": 0.5, "dsr": 0.99},
                {"name": "b", "mdd": -0.15, "calmar": 0.4, "dsr": 0.99},
            ]
        ).set_index("name")
        out = verdict(m, pbo=0.1, baseline_calmar=0.5, baseline_name="static_60_40")

        assert out.loc["static_60_40", "adopted"]
        assert "60/40" not in out.loc["static_60_40", "reason"]

    def test_baseline_exclusion_does_not_change_other_rows(self) -> None:
        """`baseline_name` 은 그 이름의 행에만 영향을 준다 — 다른 후보는
        여전히 60/40 대비 Calmar 관문을 통과해야 한다.
        """
        m = pd.DataFrame(
            [
                {"name": "static_60_40", "mdd": -0.15, "calmar": 0.5, "dsr": 0.99},
                {"name": "b", "mdd": -0.15, "calmar": 0.4, "dsr": 0.99},
            ]
        ).set_index("name")
        out = verdict(m, pbo=0.1, baseline_calmar=0.5, baseline_name="static_60_40")

        assert not out.loc["b", "adopted"]
        assert "60/40" in out.loc["b", "reason"]

    def test_without_baseline_name_behavior_is_unchanged(self) -> None:
        """`baseline_name` 을 안 넘기면(기본 `None`) 기존 동작 그대로 —
        어떤 행 이름도 예외 없이 관문을 통과해야 한다."""
        m = pd.DataFrame(
            [{"name": "static_60_40", "mdd": -0.15, "calmar": 0.5, "dsr": 0.99}]
        ).set_index("name")
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert not out.loc["static_60_40", "adopted"]
        assert "60/40" in out.loc["static_60_40", "reason"]


class TestCommonWindow:
    """9개 구성이 서로 다른 시작일을 가질 때 — BAA 는 BIL 상장 + 12개월
    모멘텀 워밍업 때문에 spy/60-40/vaa_g4 보다 늦게 시작한다. 절단 없이
    평가하면 "60/40 을 이겼다"는 채택 기준이 서로 다른 구간 위에서 비교된다.
    """

    def test_intersects_indices_of_all_series(self) -> None:
        idx_long = pd.date_range("2007-07-31", periods=230, freq="ME")
        idx_short = pd.date_range("2008-06-30", periods=219, freq="ME")

        window = common_window(
            {"spy": pd.Series(0.0, index=idx_long), "baa_bal": pd.Series(0.0, index=idx_short)}
        )

        assert window.min() == idx_short.min()
        assert window.max() == idx_long.max()
        assert len(window) == 219

    def test_truncating_to_common_window_drops_no_dates_from_the_short_series(self) -> None:
        idx_long = pd.date_range("2007-07-31", periods=230, freq="ME")
        idx_short = pd.date_range("2008-06-30", periods=219, freq="ME")
        window = common_window(
            {"a": pd.Series(0.0, index=idx_long), "b": pd.Series(0.0, index=idx_short)}
        )

        assert set(window) <= set(idx_short)


class TestEvaluateAll:
    """`run_backtest` 를 흉내내는 페이크로 아홉 구성이 서로 다른 길이의
    수익률을 낼 때 `evaluate_all` 이 공통 구간으로 자르는지 확인한다.
    실제 가격 데이터·레지스트리에 의존하지 않는다 — 그건 `scripts/run_taa.py`
    를 실제로 돌려서 확인한다.
    """

    def test_truncates_all_series_to_the_shared_window(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import opt_portfolio.taa.evaluate as ev_mod

        long_idx = pd.date_range("2007-07-31", periods=10, freq="ME")
        short_idx = pd.date_range("2007-09-30", periods=6, freq="ME")

        class _FakeSpec:
            def __init__(self, name: str) -> None:
                self.name = name

        fake_registry = {"long": _FakeSpec("long"), "short": _FakeSpec("short")}

        def _fake_run_backtest(spec: _FakeSpec, daily: object, **kw: object) -> object:
            idx = long_idx if spec.name == "long" else short_idx
            rng = np.random.default_rng(0)
            returns = pd.Series(rng.normal(0.005, 0.02, len(idx)), index=idx, name=spec.name)

            class _Out:
                pass

            out = _Out()
            out.returns = returns
            out.defensive_ratio = 0.0
            return out

        monkeypatch.setattr(ev_mod, "REGISTERED", fake_registry)
        monkeypatch.setattr(ev_mod, "TRANCHE", frozenset())
        monkeypatch.setattr(ev_mod, "MA_OVERLAY", frozenset())
        monkeypatch.setattr(ev_mod, "run_backtest", _fake_run_backtest)

        metrics, matrix = evaluate_all(pd.DataFrame(), start=None, end=None, cost_bps=10.0)

        assert len(matrix) == len(short_idx)
        assert metrics.loc["long", "months"] == float(len(short_idx))
        assert metrics.loc["short", "months"] == float(len(short_idx))
        assert not matrix.isna().any().any()

    def test_dispatches_tranche_and_ma_overlay_together_when_both_apply(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """`baa_bal_ma_tranche` 는 TRANCHE 와 MA_OVERLAY 양쪽에 다 있다.
        `if name in TRANCHE: ... elif name in MA_OVERLAY:` 로 떨어뜨리면 이
        구성은 트랜치만 받고 이평 오버레이는 받지 못한다 — config 8
        (`baa_bal_tranche`)의 조용한 중복이 된다. `run_with_tranches` 가
        `ma_overlay=True` 로 호출됐는지까지 확인해야 그 실수를 잡는다.
        """
        import opt_portfolio.taa.evaluate as ev_mod

        idx = pd.date_range("2007-07-31", periods=5, freq="ME")

        class _FakeSpec:
            def __init__(self, name: str) -> None:
                self.name = name

        fake_registry = {
            "plain": _FakeSpec("plain"),
            "ma_only": _FakeSpec("ma_only"),
            "tranche_only": _FakeSpec("tranche_only"),
            "both": _FakeSpec("both"),
        }
        calls: list[tuple[str, str, object]] = []

        def _make_out(name: str) -> object:
            returns = pd.Series(0.01, index=idx, name=name)

            class _Out:
                pass

            out = _Out()
            out.returns = returns
            out.defensive_ratio = 0.0
            return out

        def _fake_run_backtest(spec: _FakeSpec, daily: object, **kw: object) -> object:
            calls.append(("plain", spec.name, None))
            return _make_out(spec.name)

        def _fake_run_with_ma_overlay(spec: _FakeSpec, daily: object, **kw: object) -> object:
            calls.append(("ma_overlay", spec.name, None))
            return _make_out(spec.name)

        def _fake_run_with_tranches(spec: _FakeSpec, daily: object, **kw: object) -> object:
            calls.append(("tranche", spec.name, kw.get("ma_overlay")))
            return _make_out(spec.name)

        monkeypatch.setattr(ev_mod, "REGISTERED", fake_registry)
        monkeypatch.setattr(ev_mod, "TRANCHE", frozenset({"tranche_only", "both"}))
        monkeypatch.setattr(ev_mod, "MA_OVERLAY", frozenset({"ma_only", "both"}))
        monkeypatch.setattr(ev_mod, "run_backtest", _fake_run_backtest)
        monkeypatch.setattr(ev_mod, "run_with_ma_overlay", _fake_run_with_ma_overlay)
        monkeypatch.setattr(ev_mod, "run_with_tranches", _fake_run_with_tranches)

        evaluate_all(pd.DataFrame(), start=None, end=None, cost_bps=10.0)

        dispatched = {name: (kind, flag) for kind, name, flag in calls}
        assert dispatched["plain"] == ("plain", None)
        assert dispatched["ma_only"] == ("ma_overlay", None)
        assert dispatched["tranche_only"] == ("tranche", False)
        assert dispatched["both"] == ("tranche", True)
