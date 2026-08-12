"""
절대 시총 밴드 — "기관이 못 건드는 구간"의 조작적 정의.

`smallcap_bottom_pct` 는 유니버스 내 상대 백분위라, 유니버스가 바뀌면 같은
설정이 다른 종목을 고른다. S&P500 하위 20% 는 여전히 수십억 달러짜리
대형주다. 기관 접근 불가는 **절대 시총**의 문제이므로 밴드로 지정한다.

문헌 근거:
- 소외기업 효과(Arbel & Strebel 1983) 및 투자자 기반 규모 연구 — 기관
  기반이 작은 종목에서만 지속되는 아노말리가 있다.
- 반대로 Hou·Xue·Zhang(2020, RFS)은 마이크로캡이 극단 분위수를 채우면
  균등가중 수익이 과장된다고 경고한다. 그래서 하한(min_mcap_usd)과
  유동성 필터를 함께 두어 '거래 불가능한 종목'이 섞이지 않게 한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from opt_portfolio.factor.dsl.context import PanelContext
from opt_portfolio.factor.universe.filters import UniverseConfig, build_universe

DATES = pd.date_range("2024-01-01", periods=10, freq="B")
TICKERS = ["MICRO", "SMALL", "MID", "MEGA"]
MCAPS = [30e6, 300e6, 5e9, 900e9]


def _ctx() -> PanelContext:
    close = pd.DataFrame(100.0, index=DATES, columns=TICKERS)
    return PanelContext(
        daily={
            "close": close,
            "closeunadj": close,
            "volume": pd.DataFrame(1e6, index=DATES, columns=TICKERS),
            "mcap": pd.DataFrame([MCAPS] * len(DATES), index=DATES, columns=TICKERS),
        },
        quarterly={},
        meta=pd.DataFrame(index=TICKERS),
    )


def _selected(config: UniverseConfig) -> set[str]:
    mask = build_universe(_ctx(), config)
    return set(mask.columns[mask.iloc[-1]])


class TestMarketCapBand:
    def test_upper_bound_excludes_institutional_names(self) -> None:
        """기관이 사는 대형주를 잘라내는 것이 이 필터의 목적이다."""
        got = _selected(UniverseConfig(max_mcap_usd=1e9, exclude_financials=False))

        assert "MEGA" not in got and "MID" not in got
        assert {"MICRO", "SMALL"} <= got

    def test_lower_bound_excludes_untradable_dust(self) -> None:
        """하한이 없으면 체결 불가능한 초소형이 섞여 백테스트가 부풀려진다."""
        got = _selected(UniverseConfig(min_mcap_usd=100e6, exclude_financials=False))

        assert "MICRO" not in got
        assert "SMALL" in got

    def test_band_selects_only_the_middle(self) -> None:
        got = _selected(
            UniverseConfig(min_mcap_usd=100e6, max_mcap_usd=1e9, exclude_financials=False)
        )

        assert got == {"SMALL"}

    def test_no_band_keeps_everything(self) -> None:
        assert _selected(UniverseConfig(exclude_financials=False)) == set(TICKERS)

    def test_band_is_validated(self) -> None:
        with pytest.raises(ValueError, match="min_mcap_usd"):
            UniverseConfig(min_mcap_usd=1e9, max_mcap_usd=100e6)
