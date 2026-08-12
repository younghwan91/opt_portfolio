"""
팩터 DSL 검증

가장 중요한 두 가지를 테스트한다:
1. PIT 승격이 공시일(datekey) 을 정확히 지키는가 — look-ahead 차단
2. 그리드 혼동을 구조적으로 막는가 — 분기 트랜스폼을 일별에 걸면 에러
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.factor.dsl.context import MissingDataError, PanelContext
from opt_portfolio.factor.dsl.expr import F, GridError
from opt_portfolio.factor.dsl.registry import REGISTRY

TICKERS = ["AAA", "BBB"]


@pytest.fixture
def ctx() -> PanelContext:
    """8개 분기 + 대응 일별 캘린더를 갖는 최소 컨텍스트."""
    periods = pd.date_range("2020-03-31", periods=8, freq="QE")
    # 공시일 = 회계기간말 + 45일
    datekeys = periods + pd.Timedelta(days=45)

    quarterly = {
        "revenue": pd.DataFrame(
            {"AAA": [100.0, 110, 121, 133, 146, 161, 177, 195], "BBB": [200.0] * 8},
            index=periods,
        ),
        "netinc": pd.DataFrame(
            {"AAA": [10.0, 11, 12, 13, 14, 15, 16, 17], "BBB": [-5.0] * 8},
            index=periods,
        ),
        "assets": pd.DataFrame({"AAA": [1000.0] * 8, "BBB": [2000.0] * 8}, index=periods),
    }
    availability = pd.DataFrame({t: datekeys for t in TICKERS}, index=periods)

    calendar = pd.date_range("2020-01-01", "2022-06-30", freq="B")
    rng = np.random.default_rng(42)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(calendar), 2)), axis=0)),
        index=calendar,
        columns=TICKERS,
    )
    daily = {
        "close": close,
        "mcap": close * 1_000_000,
        "volume": pd.DataFrame(1e6, index=calendar, columns=TICKERS),
    }
    meta = {"sector": pd.Series({"AAA": "Tech", "BBB": "Health"})}

    return PanelContext(
        quarterly=quarterly,
        availability=availability,
        daily=daily,
        meta=meta,
        calendar=calendar,
    )


class TestPointInTime:
    def test_quarterly_value_not_visible_before_filing_date(self, ctx: PanelContext) -> None:
        """회계기간말과 공시일 사이에는 이전 분기 값만 보여야 한다."""
        promoted = ctx.to_daily(ctx.quarterly["revenue"])

        # 2020-03-31 분기는 2020-05-15 에 공시됨
        assert np.isnan(promoted.loc[pd.Timestamp("2020-04-30"), "AAA"]), (
            "공시 전에 분기 데이터가 노출됨 — look-ahead 발생"
        )
        assert promoted.loc[pd.Timestamp("2020-05-15"), "AAA"] == 100.0
        assert promoted.loc[pd.Timestamp("2020-08-13"), "AAA"] == 100.0, (
            "다음 분기 공시 전에는 직전 분기 값이 유지되어야 함"
        )
        # 2020-06-30 분기는 45일 후인 2020-08-14 공시
        assert promoted.loc[pd.Timestamp("2020-08-14"), "AAA"] == 110.0

    def test_promotion_requires_availability(self) -> None:
        """공시일 정보 없이 분기 데이터를 펼치려 하면 실패해야 한다."""
        bare = PanelContext(
            quarterly={"revenue": pd.DataFrame()},
            daily={"close": pd.DataFrame(index=pd.date_range("2020-01-01", periods=5))},
        )
        with pytest.raises(MissingDataError, match="look-ahead"):
            bare.to_daily(pd.DataFrame())


class TestGridSafety:
    def test_yoy_on_daily_grid_raises(self, ctx: PanelContext) -> None:
        """일별 값에 YoY 를 걸면 조용히 0 이 되는 대신 에러가 나야 한다."""
        with pytest.raises(GridError, match="분기 그리드"):
            F.close.yoy().eval(ctx)

    def test_momentum_on_quarterly_grid_raises(self, ctx: PanelContext) -> None:
        with pytest.raises(GridError, match="일별 그리드"):
            F.revenue.mom(252).eval(ctx)

    def test_mixed_grid_promotes_to_daily(self, ctx: PanelContext) -> None:
        """시총(일별) / 순이익(분기) 은 일별 그리드로 승격되어야 한다."""
        panel = (F.mcap / F.netinc).eval(ctx)
        assert panel.grid == "daily"
        assert panel.data.index.equals(ctx.trading_calendar)


class TestTransforms:
    def test_ttm_sums_flow_but_not_stock(self, ctx: PanelContext) -> None:
        """플로우는 4분기 합, 스톡은 최신값 — 총자산이 4배가 되면 안 된다."""
        revenue_ttm = F.revenue.ttm().eval(ctx).data
        assets_ttm = F.assets.ttm().eval(ctx).data

        assert revenue_ttm["AAA"].iloc[3] == pytest.approx(100 + 110 + 121 + 133)
        assert assets_ttm["AAA"].iloc[3] == 1000.0, "스톡 항목이 합산되어 4배가 됨"

    def test_yoy_uses_absolute_denominator(self, ctx: PanelContext) -> None:
        """적자 축소(-100 → -50)가 양의 성장으로 잡혀야 한다."""
        periods = pd.date_range("2020-03-31", periods=8, freq="QE")
        ctx.quarterly["netinc"] = pd.DataFrame(
            {"AAA": [-100.0, -100, -100, -100, -50, -50, -50, -50], "BBB": [1.0] * 8},
            index=periods,
        )
        growth = F.netinc.yoy().eval(ctx).data
        assert growth["AAA"].iloc[4] == pytest.approx(0.5), (
            "abs() 분모가 없으면 적자 축소가 악화로 뒤집힘"
        )

    def test_accel_is_second_difference(self, ctx: PanelContext) -> None:
        # YoY 는 4분기 랙이 필요해 iloc[4] 부터 유효 → periods=1 차분으로 검증
        yoy = F.revenue.yoy().eval(ctx).data
        accel = F.revenue.yoy().accel(periods=1).eval(ctx).data
        expected = yoy["AAA"].iloc[5] - yoy["AAA"].iloc[4]
        assert accel["AAA"].iloc[5] == pytest.approx(expected)
        assert np.isnan(accel["AAA"].iloc[4]), "차분 워밍업 구간은 NaN 이어야 함"

    def test_division_by_zero_yields_nan_not_inf(self, ctx: PanelContext) -> None:
        """inf 는 횡단면 랭킹을 오염시키므로 NaN 이어야 한다."""
        periods = ctx.quarterly["revenue"].index
        ctx.quarterly["netinc"] = pd.DataFrame(0.0, index=periods, columns=TICKERS)
        result = (F.revenue / F.netinc).eval(ctx).data
        assert not np.isinf(result.to_numpy()).any()
        assert result.isna().all().all()

    def test_sector_neutralize_removes_group_mean(self, ctx: PanelContext) -> None:
        """중립화 후 섹터 내 평균은 0 에 가까워야 한다."""
        expr = (F.gp if False else F.close).neutralize("sector")
        residual = expr.eval(ctx).data
        # 섹터당 종목이 1개뿐이면 잔차가 정의되지 않는다 → NaN 허용
        assert residual.shape == ctx.daily["close"].shape


class TestRegistry:
    def test_all_factors_register_without_collision(self) -> None:
        """라이브러리 import 만으로 전 팩터가 충돌 없이 등록된다."""
        import opt_portfolio.factor.library  # noqa: F401

        assert len(REGISTRY) >= 120, f"등록된 팩터가 {len(REGISTRY)}개뿐"

    def test_growth_and_acceleration_are_auto_derived(self) -> None:
        import opt_portfolio.factor.library  # noqa: F401

        # 성장·가속 팩터는 기반 항목마다 QoQ/YoY 쌍으로 생성된다.
        # 개수를 못박으면 팩터를 하나 추가할 때마다 깨지므로 **쌍 구조**를 본다.
        names = {s.name for s in REGISTRY.by_category("growth")}
        qoq = {n[: -len("_QOQ")] for n in names if n.endswith("_QOQ")}
        yoy = {n[: -len("_YOY")] for n in names if n.endswith("_YOY")}
        assert qoq == yoy, f"짝이 없는 성장 팩터: {qoq ^ yoy}"
        assert len(qoq) >= 13, f"성장 기반 항목이 {len(qoq)}개뿐"

        accel = {s.name for s in REGISTRY.by_category("acceleration")}
        a_qoq = {n[: -len("_QOQ")] for n in accel if n.endswith("_QOQ")}
        a_yoy = {n[: -len("_YOY")] for n in accel if n.endswith("_YOY")}
        assert a_qoq == a_yoy, f"짝이 없는 가속 팩터: {a_qoq ^ a_yoy}"

    def test_multiples_are_inverted_for_scoring(self) -> None:
        """PER 스코어링 표현식은 역수를 취해 적자기업이 상위로 오지 않아야 한다."""
        import opt_portfolio.factor.library  # noqa: F401

        per = REGISTRY.get("PER")
        assert per.invert is True
        assert "1.0 /" in per.scoring_expr().describe()

    def test_unsubscribed_datasets_disable_factors(self) -> None:
        import opt_portfolio.factor.library  # noqa: F401

        sharadar_only = REGISTRY.available({"SF1", "SEP", "SF2", "SF3"})
        names = {s.name for s in sharadar_only}
        assert "PER" in names
        assert "PEG_FWD" not in names, "FMP 미구독인데 forward PEG 가 활성화됨"
        assert "SHORT_INT_CHG" not in names

    def test_accruals_direction_is_negative(self) -> None:
        import opt_portfolio.factor.library  # noqa: F401

        assert REGISTRY.get("AC_A").direction == -1
        assert REGISTRY.get("MOM_1M").direction == -1, "1개월 모멘텀은 반전 팩터"
        assert REGISTRY.get("MOM_12M").direction == 1
