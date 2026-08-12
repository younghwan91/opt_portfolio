"""프로덕션 레이어 검증 — 스토어 PIT, 소스별 공시일, 유니버스, 파이프라인 E2E, CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.backtest.engine import BacktestConfig
from opt_portfolio.factor.data.provider import validate_pit_frame
from opt_portfolio.factor.data.sharadar import SharadarProvider, TruncatedDataError
from opt_portfolio.factor.data.store import PITStore
from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.optimize.walkforward import run_walk_forward
from opt_portfolio.factor.pipeline import FactorPipeline, StrategyConfig
from opt_portfolio.factor.universe.filters import UniverseConfig, build_universe
from tests.factor.conftest import BENCH, NORMAL, QUARTERS, populate


class TestStore:
    def test_first_filing_wins_over_restatement(self) -> None:
        """같은 (ticker, 분기) 재공시는 최초 datekey 값을 유지해야 한다."""
        with PITStore(":memory:") as store:
            first = pd.DataFrame(
                [
                    {
                        "ticker": "A",
                        "calendardate": "2024-03-31",
                        "datekey": "2024-05-10",
                        "netinc": 100.0,
                    }
                ]
            )
            restated = pd.DataFrame(
                [
                    {
                        "ticker": "A",
                        "calendardate": "2024-03-31",
                        "datekey": "2024-08-01",
                        "netinc": 120.0,
                    }
                ]
            )
            assert store.upsert_fundamentals(first) == 1
            assert store.upsert_fundamentals(restated) == 0  # 무시됨
            ctx = store.build_context()
            assert ctx.quarterly["netinc"].iloc[0, 0] == 100.0

    def test_upsert_is_idempotent(self, synth_store: PITStore) -> None:
        rows = pd.DataFrame(
            [
                {
                    "ticker": "T00",
                    "calendardate": QUARTERS[0],
                    "datekey": QUARTERS[0] + pd.Timedelta(days=45),
                    "dimension": "ARQ",
                    "netinc": 999.0,
                }
            ]
        )
        assert synth_store.upsert_fundamentals(rows) == 0

    def test_coverage_reports_all_tables(self, synth_store: PITStore) -> None:
        cov = synth_store.coverage().set_index("table")
        assert cov.loc["fundamentals", "rows"] > 0
        assert cov.loc["prices", "tickers"] == len(NORMAL) + 6  # 특수 5 + SPY
        assert cov.loc["institutions", "rows"] > 0

    def test_context_is_point_in_time(self, synth_ctx) -> None:
        """분기값은 datekey(+45일) 전에 일별 그리드에 나타나면 안 된다."""
        daily = synth_ctx.eval_daily(F.netinc)
        q0 = QUARTERS[3]  # 2017-12-31 → 공시 2018-02-14
        before = daily.loc[: q0 + pd.Timedelta(days=40), "T00"]
        after = daily.loc[q0 + pd.Timedelta(days=46) :, "T00"]
        # 직전 분기(2017-09-30, 공시 11-14)값만 보이고, 새 분기값은 공시 후 반영
        assert not before.empty
        assert after.iloc[0] != before.iloc[-1]

    def test_source_specific_availability(self, synth_ctx) -> None:
        """13F(+75일) 필드는 SF1(+45일)보다 늦게 보여야 한다."""
        sf1_daily = synth_ctx.eval_daily(F.netinc)
        sf3_daily = synth_ctx.eval_daily(F.inst_shares)
        q = QUARTERS[4]  # 2018-03-31
        probe = q + pd.Timedelta(days=60)  # 공시 사이 시점 (45 < 60 < 75)
        probe = sf1_daily.index[sf1_daily.index.searchsorted(probe)]

        sf1_at_probe = sf1_daily.loc[probe, "T00"]
        sf3_at_probe = sf3_daily.loc[probe, "T00"]
        # SF1 은 이미 새 분기 공시됨 — 값 존재
        assert np.isfinite(sf1_at_probe)
        # SF3 는 직전 분기값 (또는 NaN) — 새 분기 inst_shares 는 아직
        new_inst = 400_000.0 + 1_000.0 * 4
        assert sf3_at_probe != new_inst

    def test_mixed_source_expression_uses_latest_availability(self, synth_ctx) -> None:
        """SF1×SF3 혼합 팩터는 늦은 쪽(+75일) 공시일을 따라야 한다."""
        mixed = synth_ctx.eval_daily(F.inst_shares / F.sharesbas)
        q = QUARTERS[4]
        probe_60 = mixed.index[mixed.index.searchsorted(q + pd.Timedelta(days=60))]
        probe_80 = mixed.index[mixed.index.searchsorted(q + pd.Timedelta(days=80))]
        new_ratio = (400_000.0 + 1_000.0 * 4) / 1_000_000.0
        assert mixed.loc[probe_60, "T00"] != pytest.approx(new_ratio)
        assert mixed.loc[probe_80, "T00"] == pytest.approx(new_ratio)


class TestProviderAdapters:
    def test_pit_contract_rejects_impossible_dates(self) -> None:
        # reportperiod 가 있으면 그것으로 엄격 검증
        bad = pd.DataFrame(
            [
                {
                    "ticker": "A",
                    "calendardate": "2024-06-30",
                    "reportperiod": "2024-06-30",
                    "datekey": "2024-05-01",
                }
            ]
        )
        with pytest.raises(ValueError, match="PIT 계약 위반"):
            validate_pit_frame(bad)
        # reportperiod 없으면 92일(달력 스냅 허용치) 초과 조기만 거부
        bad2 = pd.DataFrame(
            [{"ticker": "A", "calendardate": "2024-06-30", "datekey": "2024-01-01"}]
        )
        with pytest.raises(ValueError, match="PIT 계약 위반"):
            validate_pit_frame(bad2)

    def test_pit_contract_allows_nonstandard_fiscal_year(self) -> None:
        """NKE(5월 결산): datekey 가 calendardate 직전이어도 정상 (실데이터 케이스)."""
        nke = pd.DataFrame(
            [
                {
                    "ticker": "NKE",
                    "calendardate": "2025-12-31",
                    "reportperiod": "2025-11-30",
                    "datekey": "2025-12-30",
                }
            ]
        )
        validate_pit_frame(nke)  # 예외 없어야 함

    def test_sharadar_pagination_and_normalization(self) -> None:
        pages = [
            {
                "datatable": {
                    "columns": [
                        {"name": c}
                        for c in [
                            "ticker",
                            "calendardate",
                            "datekey",
                            "dimension",
                            "revenue",
                            "netinc",
                            "debtusd",
                        ]
                    ],
                    "data": [["AAPL", "2024-03-31", "2024-05-02", "ARQ", 90.0, 23.0, 100.0]],
                },
                "meta": {"next_cursor_id": "abc"},
            },
            {
                "datatable": {
                    "columns": [
                        {"name": c}
                        for c in [
                            "ticker",
                            "calendardate",
                            "datekey",
                            "dimension",
                            "revenue",
                            "netinc",
                            "debtusd",
                        ]
                    ],
                    "data": [["MSFT", "2024-03-31", "2024-04-25", "ARQ", 61.0, 21.0, 50.0]],
                },
                "meta": {"next_cursor_id": None},
            },
        ]
        calls = []

        def fake_get(url: str, params: dict) -> dict:
            calls.append(params.get("qopts.cursor_id"))
            return pages[len(calls) - 1]

        provider = SharadarProvider(api_key="test", get_json=fake_get, api="ndl")
        chunks = list(provider.fundamentals())
        assert len(chunks) == 2
        assert calls == [None, "abc"]  # 커서 전달 확인
        assert "debt" in chunks[0].columns  # debtusd → debt 정규화

    def test_insider_aggregation_sums_quarter(self) -> None:
        from opt_portfolio.factor.data.sharadar import _aggregate_insiders

        raw = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "filingdate": ["2024-01-15", "2024-02-20", "2024-05-01"],
                "transactionshares": [1000.0, -300.0, 500.0],
            }
        )
        out = _aggregate_insiders(raw)
        q1 = out[out["calendardate"] == pd.Timestamp("2024-03-31")]
        assert q1["insider_net_shares"].iloc[0] == 700.0  # 1000 − 300
        # 분기 합계는 분기가 끝나야 확정 — datekey = 분기말 + 3일
        assert q1["datekey"].iloc[0] == pd.Timestamp("2024-04-03")


class TestUniverse:
    def test_special_tickers_are_excluded(self, synth_ctx) -> None:
        config = UniverseConfig(min_adv_usd=0.0)  # ADV 워밍업 영향 제거
        mask = build_universe(synth_ctx, config)
        last = mask.iloc[-1]
        assert not last["FINCO"], "금융주가 유니버스에 남아 있음"
        assert not last["CHINACO"], "중국기업이 유니버스에 남아 있음"
        assert not last["OILLP"], "PTP(L.P.)가 유니버스에 남아 있음"
        assert not last["PENNY"], "페니스톡($2)이 유니버스에 남아 있음"
        assert last["T00"], "정상 종목이 제외됨"

    def test_deficit_filter_is_point_in_time(self, synth_ctx) -> None:
        config = UniverseConfig(exclude_deficit_ttm=True, min_adv_usd=0.0, min_price_usd=0.0)
        mask = build_universe(synth_ctx, config)
        assert not mask.iloc[-1]["LOSSCO"], "만년 적자 기업이 남아 있음"
        # 첫 TTM 이 공시되기 전에는 흑자 기업도 편입 불가 (검증 불가 = 제외)
        assert not mask.iloc[0]["T00"]
        assert mask.iloc[-1]["T00"]

    def test_wics_industry_filter(self, synth_ctx) -> None:
        config = UniverseConfig(wics_industries=("소프트웨어",), min_adv_usd=0.0)
        mask = build_universe(synth_ctx, config)
        assert mask.iloc[-1]["T00"]  # Software - Application → 소프트웨어
        assert not mask.iloc[-1]["FINCO"]


class TestPipelineE2E:
    @pytest.fixture(scope="class")
    def pipeline(self, synth_ctx) -> FactorPipeline:
        import opt_portfolio.factor.library  # noqa: F401

        return FactorPipeline(synth_ctx)

    @pytest.fixture(scope="class")
    def strategy(self) -> StrategyConfig:
        return StrategyConfig(
            factors=("PER_TTM", "GP_A", "MOM_12_1"),
            universe=UniverseConfig(min_adv_usd=0.0, exclude_distressed=False),
            backtest=BacktestConfig(n_stocks=5, rebalance="ME"),
            benchmark=BENCH,
        )

    def test_full_backtest_runs(self, pipeline, strategy) -> None:
        result = pipeline.run(strategy, start="2019-06-01")
        stats = result.stats()
        assert stats["n_rebalances"] >= 24
        assert np.isfinite(stats["sharpe"])
        # 특수 종목은 어떤 시점에도 보유되지 않아야 한다
        held = set(result.holdings.columns[(result.holdings != 0).any()])
        assert held.isdisjoint({"FINCO", "CHINACO", "OILLP", "PENNY", BENCH})

    def test_n_stocks_config_respected(self, pipeline, strategy) -> None:
        from dataclasses import replace

        small = replace(strategy, backtest=BacktestConfig(n_stocks=3, rebalance="ME"))
        result = pipeline.run(small, start="2019-06-01")
        max_held = (result.holdings != 0).sum(axis=1).max()
        assert max_held <= 3

    def test_evaluator_rejects_unknown_params(self, pipeline, strategy) -> None:
        evaluate = pipeline.evaluator(strategy)
        with pytest.raises(KeyError, match="알 수 없는 PO 파라미터"):
            evaluate(
                {"n_stcoks": 10},  # 오타
                pd.Timestamp("2019-06-01"),
                pd.Timestamp("2020-06-01"),
            )

    def test_walk_forward_integration(self, pipeline, strategy) -> None:
        """스토어 → 파이프라인 → PO 전 경로가 실제로 이어지는가."""
        result = run_walk_forward(
            pipeline.evaluator(strategy),
            {"n_stocks": ("int", 3, 8), "rebalance": ("cat", ["ME", "QE"])},
            pd.DatetimeIndex(pipeline.close.index),
            method="random",
            n_trials_per_fold=4,
            min_train_years=1.5,
            test_months=6,
            embargo_days=21,
            seed=1,
        )
        assert len(result.folds) >= 2
        assert len(result.oos_returns.dropna()) > 100
        assert result.n_trials_total == 4 * len(result.folds)
        assert 0.0 <= result.deflated_sharpe() <= 1.0


class TestCLI:
    def test_factors_catalog(self, capsys) -> None:
        from opt_portfolio.factor.cli import main

        assert main(["factors", "--category", "growth"]) == 0
        out = capsys.readouterr().out
        assert "REVENUE_GROWTH_YOY" in out
        assert "개 팩터" in out

    def test_backtest_command_end_to_end(self, tmp_path, capsys) -> None:
        """파일 스토어 + JSON 설정으로 CLI 백테스트가 완주하는가."""
        db = tmp_path / "test.duckdb"
        with PITStore(db) as store:
            populate(store)

        config = {
            "factors": ["PER_TTM", "GP_A", "MOM_12_1"],
            "universe": {"min_adv_usd": 0.0, "exclude_distressed": False},
            "backtest": {"n_stocks": 5, "rebalance": "ME"},
            "benchmark": BENCH,
        }
        config_path = tmp_path / "strategy.json"
        config_path.write_text(json.dumps(config))

        from opt_portfolio.factor.cli import main

        code = main(
            ["backtest", "--store", str(db), "--config", str(config_path), "--start", "2019-06-01"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "sharpe" in out
        assert "공식 성과가 아닙니다" in out  # 규율 경고 노출 확인

    def test_strategy_config_rejects_typos(self, tmp_path) -> None:
        from opt_portfolio.factor.cli import load_strategy

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"factors": ["PER"], "univrese": {}}))
        with pytest.raises(SystemExit, match="없는 설정 키"):
            load_strategy(bad)


class TestDirectAPI:
    """sharadar.com 직판 REST — 티커 청크 + 날짜 역방향 마칭."""

    def test_marches_forward_with_from_not_backward_with_to(self) -> None:
        """
        ticker 필터가 있으면 sort=date.asc 가 선택까지 지배해 가장 오래된
        구간이 먼저 온다. 따라서 from 을 올려 전진해야 한다 — to 를 내리면
        첫 페이지에서 즉시 끝난다 (DAILY 가 5년 중 2년만 들어왔던 회귀).
        """
        page1 = {
            "count": 2,
            "data": [
                {
                    "ticker": "AAPL",
                    "calendardate": "2023-12-31",
                    "date": "2024-02-01",
                    "dimension": "ARQ",
                    "revenue": "119.0",
                },
                {
                    "ticker": "AAPL",
                    "calendardate": "2024-03-31",
                    "date": "2024-05-02",
                    "dimension": "ARQ",
                    "revenue": "90.0",
                },
            ],
        }
        page2 = {
            "count": 1,
            "data": [
                {
                    "ticker": "AAPL",
                    "calendardate": "2024-06-30",
                    "date": "2024-08-01",
                    "dimension": "ARQ",
                    "revenue": "85.0",
                },
            ],
        }
        calls: list[dict] = []

        def fake_get(url: str, params: dict) -> dict:
            calls.append(dict(params))
            assert "api.sharadar.com/v1.0/data/fundamentals" in url
            return page1 if len(calls) == 1 else page2

        provider = SharadarProvider(api_key="test", get_json=fake_get, api="direct", page_size=2)
        chunks = list(provider.fundamentals(tickers=["AAPL"]))
        assert len(chunks) == 2
        assert calls[0]["sort"] == "date.asc"
        assert "from" not in calls[0]  # 첫 페이지는 가장 과거부터
        assert "to" not in calls[1]  # to 로 내려가면 안 된다
        assert calls[1]["from"] == "2024-05-02"  # 1페이지 최신 date 에서 이어받기
        assert "datekey" in chunks[0].columns

    def test_raises_when_single_date_exceeds_page(self) -> None:
        """
        한 날짜가 페이지를 꽉 채우면 전진 불가 — 무한루프를 막되 조용히 끝내지 않는다.

        여기서 그냥 return 하면 해당 날짜 이후 구간이 통째로 누락된 채
        '성공'으로 보고된다. 부분 데이터로 백테스트가 돌아가는 것을 막기 위해
        TruncatedDataError 로 즉시 실패한다.
        """
        same_day = {
            "count": 2,
            "data": [
                {"ticker": "A", "date": "2024-01-02", "closeadj": 1.0},
                {"ticker": "B", "date": "2024-01-02", "closeadj": 2.0},
            ],
        }
        calls = []

        def fake_get(url: str, params: dict) -> dict:
            calls.append(params)
            return same_day

        provider = SharadarProvider(api_key="t", get_json=fake_get, api="direct", page_size=2)
        with pytest.raises(TruncatedDataError, match="페이지 전진 불가"):
            list(provider.prices(tickers=["A", "B"]))
        assert len(calls) <= 3  # 무한루프가 아니라 즉시 감지

    def test_raises_when_page_cap_reached(self) -> None:
        """
        페이지 상한(500)까지 계속 꽉 찬 페이지가 오면 뒷구간이 남아 있다는 뜻이다.

        상한은 무한루프 방지장치일 뿐 '데이터를 다 받았다'는 신호가 아니므로,
        여기서 조용히 끝내면 절단이 성공으로 둔갑한다.
        """
        day = iter(pd.date_range("2000-01-03", periods=600, freq="D"))

        def fake_get(url: str, params: dict) -> dict:
            d = next(day).strftime("%Y-%m-%d")
            # 항상 page_size 만큼 꽉 채워 돌려준다 = 아직 남았다는 신호
            return {
                "count": 2,
                "data": [
                    {"ticker": "A", "date": d, "closeadj": 1.0},
                    {"ticker": "B", "date": d, "closeadj": 2.0},
                ],
            }

        provider = SharadarProvider(api_key="t", get_json=fake_get, api="direct", page_size=2)
        with pytest.raises(TruncatedDataError, match="페이지 상한"):
            list(provider.prices(tickers=["A", "B"]))

    def test_chunks_tickers_to_stay_under_page_limit(self) -> None:
        """SEP 은 티커당 행수가 커서 소규모 청크로 쪼개 요청해야 한다."""
        calls: list[dict] = []

        def fake_get(url: str, params: dict) -> dict:
            calls.append(dict(params))
            return {
                "count": 1,
                "data": [{"ticker": "X", "date": "2024-01-02", "closeadj": 1.0, "volume": 1.0}],
            }

        provider = SharadarProvider(api_key="t", get_json=fake_get, api="direct")
        names = [f"T{i:02d}" for i in range(12)]
        list(provider.prices(tickers=names))
        # SEP 청크 크기 5 → 12종목이면 3회 요청, 각 요청의 티커는 5개 이하
        assert len(calls) == 3
        for call in calls:
            assert len(call["ticker"].split(",")) <= 5

    def test_parses_columns_data_payload_shape(self) -> None:
        from opt_portfolio.factor.data.sharadar import _parse_direct_payload

        frame = _parse_direct_payload(
            {"columns": ["ticker", "datekey"], "data": [["AAPL", "2024-05-02"]]}
        )
        assert list(frame.columns) == ["ticker", "datekey"]
        assert frame.iloc[0, 0] == "AAPL"


class TestReport:
    def test_tearsheet_renders_gates_and_charts(self, synth_ctx, tmp_path) -> None:
        import opt_portfolio.factor.library  # noqa: F401
        from opt_portfolio.factor.pipeline import FactorPipeline
        from opt_portfolio.factor.report import render_tearsheet

        pipeline = FactorPipeline(synth_ctx)
        strategy = StrategyConfig(
            factors=("PER_TTM", "GP_A", "MOM_12_1"),
            universe=UniverseConfig(min_adv_usd=0.0, exclude_distressed=False),
            backtest=BacktestConfig(n_stocks=5, rebalance="ME"),
            benchmark=BENCH,
        )
        result = pipeline.run(strategy, start="2019-06-01")
        html = render_tearsheet(result, title="테스트 전략", config_summary={"종목": "5"})
        # 판정 게이트가 최상단 규율로 존재하고, 미실행 상태가 명시된다
        assert "판정 게이트" in html
        assert "미실행" in html
        assert "참고용" in html  # walk-forward 없으면 공식 배지 금지
        assert "사용 금지" in html
        # 차트 두 종 + 히트맵
        assert html.count("<svg") >= 2
        assert "월별 수익률" in html
        (tmp_path / "t.html").write_text(html)  # 파일로 쓰기 가능한지
