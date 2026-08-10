"""
`configs/` 의 즉시 실행용 설정이 실제로 열리는지 검증한다.

이 설정들의 존재 이유는 "구독한 날 바로 optimize 를 돌린다"이다.
설정이 코드와 어긋나 그날 실패하면 존재 의미가 없으므로, 팩터 이름 오타나
스키마 변경이 여기서 먼저 깨지게 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opt_portfolio.factor.cli import load_space, load_strategy
from opt_portfolio.factor.optimize.search import sample_params

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="module")
def strategy():
    import opt_portfolio.factor.library  # noqa: F401  (레지스트리 등록)

    return load_strategy(CONFIG_DIR / "strategy.json")


class TestShippedStrategy:
    def test_loads(self, strategy) -> None:
        assert strategy.factors, "팩터 목록이 비었다"

    def test_every_factor_name_exists(self, strategy) -> None:
        """오타 난 팩터 이름은 resolved_factors() 에서 조용히 빠진다."""
        assert len(strategy.resolved_factors()) == len(strategy.factors), (
            "레지스트리에 없거나 구독 데이터셋으로 계산 불가한 팩터가 있다"
        )

    def test_covers_multiple_categories(self, strategy) -> None:
        """한 카테고리에 몰린 조합은 멀티팩터가 아니다."""
        categories = {s.category for s in strategy.resolved_factors()}
        assert len(categories) >= 4, f"카테고리가 {categories} 뿐이다"


class TestShippedSpace:
    def test_space_is_samplable(self) -> None:
        space = load_space(CONFIG_DIR / "space.json")
        params = sample_params(space, np.random.default_rng(0))

        assert set(params) == set(space), "탐색 공간 키와 샘플 키가 다르다"

    def test_space_keys_are_accepted_by_pipeline(self) -> None:
        """
        pipeline.evaluator 는 모르는 키에 KeyError 를 낸다 (조용한 오타 방지).
        여기 키가 그 화이트리스트 안에 있어야 optimize 가 첫 폴드에서 죽지 않는다.
        """
        from opt_portfolio.factor.pipeline import _BT_KEYS

        space = load_space(CONFIG_DIR / "space.json")
        unknown = {k for k in space if k not in _BT_KEYS and not k.startswith("w_")}

        assert not unknown, f"파이프라인이 모르는 탐색 키: {sorted(unknown)}"
