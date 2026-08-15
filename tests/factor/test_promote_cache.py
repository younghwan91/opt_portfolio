"""
분기→일별 승격 캐시의 크기 제한.

왜 필요한가 (실측): 풀 히스토리 walk-forward 가 OOM 으로 두 번 죽었다.
원인은 폴드마다 새는 것이 아니라 **캐시 하나가 무제한으로 자라는 것**이었다.

    데이터 로드          RSS 2.70GB   승격 캐시 0개
    evaluator 구성       RSS 5.66GB   승격 캐시 5개 / 1,811MB
    폴드 1 (팩터 20개)   RSS 10.4GB

승격 프레임 하나가 362MB 다 — 9,000 거래일 × 6,895종목. 팩터를 늘릴수록
선형으로 쌓이므로, 팩터 수가 곧 메모리 한계가 된다. 캐시가 실제로 값을
하는 것은 한 표현식 안의 재사용뿐이라(파이프라인은 승격 결과를 신호일
그리드로 줄인 직후 버린다) 작은 LRU 로 충분하다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from opt_portfolio.factor.dsl.context import PROMOTE_CACHE_SIZE, PanelContext


def _ctx(n_frames: int) -> tuple[PanelContext, list[pd.DataFrame]]:
    cal = pd.bdate_range("2020-01-01", periods=40)
    qdates = pd.to_datetime(["2019-12-31", "2020-03-31"])
    cols = ["AAA", "BBB"]
    avail = pd.DataFrame(
        [[pd.Timestamp("2020-02-15")] * 2, [pd.Timestamp("2020-05-15")] * 2],
        index=qdates,
        columns=cols,
    )
    frames = [
        pd.DataFrame(np.full((2, 2), float(i)), index=qdates, columns=cols)
        for i in range(n_frames)
    ]
    ctx = PanelContext(
        quarterly={f"f{i}": f for i, f in enumerate(frames)},
        availability=avail,
        daily={"close": pd.DataFrame(1.0, index=cal, columns=cols)},
        calendar=cal,
    )
    return ctx, frames


class TestPromoteCacheBound:
    def test_cache_does_not_grow_without_limit(self) -> None:
        ctx, frames = _ctx(PROMOTE_CACHE_SIZE + 5)

        for frame in frames:
            ctx.to_daily(frame)

        assert len(ctx._promote_cache) <= PROMOTE_CACHE_SIZE

    def test_recently_used_frame_survives_eviction(self) -> None:
        """LRU 여야 한다 — 방금 쓴 것을 버리면 캐시가 오히려 해가 된다."""
        ctx, frames = _ctx(PROMOTE_CACHE_SIZE + 1)
        hot = frames[0]

        for frame in frames[:PROMOTE_CACHE_SIZE]:
            ctx.to_daily(frame)
        ctx.to_daily(hot)  # 재사용 → 최신으로 올라간다
        ctx.to_daily(frames[-1])  # 새 항목 → 하나가 밀려난다

        assert (id(hot), id(ctx.availability)) in ctx._promote_cache

    def test_evicted_frame_still_promotes_correctly(self) -> None:
        """
        캐시는 성능 장치일 뿐이다. 밀려난 프레임을 다시 요청했을 때 값이
        달라지면 그건 캐시가 아니라 상태다.
        """
        ctx, frames = _ctx(PROMOTE_CACHE_SIZE + 3)
        first = ctx.to_daily(frames[0])

        for frame in frames[1:]:
            ctx.to_daily(frame)
        again = ctx.to_daily(frames[0])

        pd.testing.assert_frame_equal(first, again)
