"""
섹터 비중 상한 — 팩터는 섹터 중립화하는데 포트폴리오는 그렇지 않았다.

`neutralize=("sector",)` 는 **스코어**에서 섹터 효과를 뺀다. 그런데 상위 N을
고르고 나면 그 결과가 한 섹터에 몰릴 수 있고, 아무도 그걸 보지 않았다.
실측: 채택 전략의 현재 보유 19종목 중 6종목(32%)이 Technology 였다.

섹터 쏠림은 팩터 베팅이 아니라 **의도하지 않은 매크로 베팅**이다.
금리가 움직이면 그 하나로 포트폴리오가 결정된다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from opt_portfolio.factor.portfolio.weights import cap_sector_weights


@pytest.fixture
def concentrated():
    weights = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy", "E": "Health"})
    return weights, sectors


class TestCapSectorWeights:
    def test_caps_the_dominant_sector(self, concentrated) -> None:
        weights, sectors = concentrated

        capped = cap_sector_weights(weights, sectors, max_sector_weight=0.4)

        tech = capped[sectors == "Tech"].sum()
        assert tech == pytest.approx(0.4)

    def test_still_sums_to_one(self, concentrated) -> None:
        weights, sectors = concentrated

        capped = cap_sector_weights(weights, sectors, max_sector_weight=0.4)

        assert capped.sum() == pytest.approx(1.0)

    def test_excess_goes_to_other_sectors(self, concentrated) -> None:
        weights, sectors = concentrated

        capped = cap_sector_weights(weights, sectors, max_sector_weight=0.4)

        assert capped["D"] > weights["D"], "잘린 비중이 다른 섹터로 가지 않았다"

    def test_within_sector_proportions_preserved(self) -> None:
        """섹터를 자르되 그 안의 상대 비중은 유지한다 — 스코어 순서를 뒤집지 않는다."""
        weights = pd.Series({"A": 0.4, "B": 0.2, "C": 0.4})
        sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Energy"})

        capped = cap_sector_weights(weights, sectors, max_sector_weight=0.3)

        assert capped["A"] / capped["B"] == pytest.approx(weights["A"] / weights["B"])

    def test_no_change_when_already_within_cap(self, concentrated) -> None:
        weights, sectors = concentrated

        capped = cap_sector_weights(weights, sectors, max_sector_weight=0.9)

        pd.testing.assert_series_equal(capped, weights, check_names=False)

    def test_missing_sector_is_its_own_bucket(self) -> None:
        """섹터를 모르는 종목을 한 덩어리로 묶어 잘라내면 안 된다."""
        weights = pd.Series({"A": 0.5, "B": 0.5})
        sectors = pd.Series({"A": None, "B": None})

        capped = cap_sector_weights(weights, sectors, max_sector_weight=0.4)

        assert capped.sum() == pytest.approx(1.0)
        assert capped["A"] == pytest.approx(0.5)

    def test_infeasible_cap_falls_back(self) -> None:
        """섹터 수 × 상한 < 1 이면 만족 불가 — 조용히 이상한 값을 내지 않는다."""
        weights = pd.Series({"A": 0.5, "B": 0.5})
        sectors = pd.Series({"A": "Tech", "B": "Tech"})

        capped = cap_sector_weights(weights, sectors, max_sector_weight=0.3)

        assert capped.sum() == pytest.approx(1.0)
