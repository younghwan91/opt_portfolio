"""
평가 컨텍스트 — Point-in-Time 보장의 실행 지점

표현식은 원시 테이블에 직접 접근할 수 없다. 오직 PanelContext 를 통해서만
데이터를 얻으며, 컨텍스트가 공시일(datekey) 기준 정렬을 강제한다.
따라서 look-ahead 는 개발자의 규율이 아니라 구조로 차단된다.

퀀트 관점:
- 분기 데이터는 `calendardate`(회계기간말) 인덱스로 보관하되, 일별 승격 시에는
  반드시 `datekey`(공시일) 를 기준으로 매핑한다. 결산일과 공시일 사이는
  최대 90일이며, 이 구간을 잘못 다루면 백테스트가 조용히 미래를 본다.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd

from opt_portfolio.factor.data.schema import FieldKind, get_field
from opt_portfolio.factor.dsl.expr import Panel

logger = logging.getLogger(__name__)

#: 승격 캐시에 남길 프레임 수. 풀 히스토리에서 승격 프레임 하나가 362MB 다
#: (9,000 거래일 × 6,895종목). 무제한이면 팩터 20개를 평가하는 것만으로
#: 수 GB 가 쌓여 15GB 머신이 OOM 으로 죽는다 (2026-08-15 실측).
#:
#: 캐시의 실질적 값어치는 **한 표현식 안의 재사용**이다 — 파이프라인은 승격
#: 결과를 신호일 그리드로 줄인 직후 버리므로 호출 사이에 다시 쓰지 않는다.
#: 그래서 작은 LRU 로 충분하다.
PROMOTE_CACHE_SIZE = 4


@dataclass
class PanelContext:
    """
    팩터 평가에 필요한 모든 데이터를 담는 컨테이너.

    Args:
        quarterly: {필드명: (calendardate × ticker) 프레임}
        availability: (calendardate × ticker) 프레임. 각 셀의 공시일(datekey).
            분기 데이터를 일별로 승격할 때의 기본 기준.
        availability_by_source: {소스명("SF1"/"SF3"/…): 공시일 프레임}.
            소스마다 공시 지연이 다를 때 (13F 는 +45일) 필드별로 올바른
            공시일을 쓰기 위한 오버라이드. 없는 소스는 기본값으로 폴백.
        daily: {필드명: (거래일 × ticker) 프레임}
        meta: {필드명: (ticker → 값) 시리즈}. 섹터·소재지 등 저빈도 속성.
        calendar: 일별 평가 캘린더 (거래일). 미지정 시 daily 프레임에서 추론.
    """

    quarterly: dict[str, pd.DataFrame] = dc_field(default_factory=dict)
    availability: pd.DataFrame | None = None
    daily: dict[str, pd.DataFrame] = dc_field(default_factory=dict)
    meta: dict[str, pd.Series] = dc_field(default_factory=dict)
    calendar: pd.DatetimeIndex | None = None
    availability_by_source: dict[str, pd.DataFrame] = dc_field(default_factory=dict)

    # (id(frame), id(avail)) → (원본 참조들, 승격 결과). 원본을 함께 보관해
    # GC 로 id 가 재사용되면서 엉뚱한 캐시가 히트하는 것을 막는다.
    # PROMOTE_CACHE_SIZE 개로 제한되는 LRU — 크기 제한이 없으면 메모리가 샌다.
    _promote_cache: OrderedDict[
        tuple[int, int], tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]
    ] = dc_field(default_factory=OrderedDict, repr=False, compare=False)

    # ------------------------------------------------------------------ 필드 접근
    def field(self, name: str) -> Panel:
        spec = get_field(name)

        if spec.grid == "quarterly":
            if name not in self.quarterly:
                raise MissingDataError(
                    f"분기 필드 '{name}' 이(가) 컨텍스트에 없습니다. "
                    f"프로바이더가 이 필드를 제공하는지 확인하세요."
                )
            return Panel(self.quarterly[name], "quarterly", self._avail_for(spec.source))

        if name in self.daily:
            return Panel(self.daily[name], "daily")

        if name in self.meta:
            return Panel(self._broadcast_meta(self.meta[name]), "daily")

        raise MissingDataError(
            f"일별 필드 '{name}' 이(가) 컨텍스트에 없습니다. "
            f"보유 필드: {sorted(self.daily) + sorted(self.meta)}"
        )

    def field_kind(self, name: str) -> FieldKind:
        return get_field(name).kind

    def _avail_for(self, source: str) -> pd.DataFrame | None:
        """소스별 공시일 프레임 — 오버라이드 없으면 기본값."""
        return self.availability_by_source.get(source, self.availability)

    # -------------------------------------------------------------- PIT 승격
    def to_daily(self, quarterly: pd.DataFrame, avail: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        분기 프레임을 일별 캘린더로 승격한다.

        각 (일자 d, 종목 t) 에는 `공시일 <= d` 를 만족하는 가장 최근 분기값이
        들어간다. 공시 전 분기 데이터는 절대 노출되지 않는다.

        Args:
            avail: 이 프레임에 적용할 셀별 공시일. 표현식 평가 경로에서는
                Panel.avail (소스별 공시일이 BinOp 를 거치며 max 합성된 것)
                이 넘어온다. None 이면 컨텍스트 기본값.
        """
        avail_frame = avail if avail is not None else self.availability
        if avail_frame is None:
            raise MissingDataError(
                "availability(datekey) 프레임이 없어 PIT 승격을 할 수 없습니다. "
                "공시일 없이 분기 데이터를 일별로 펼치면 look-ahead 가 발생합니다."
            )

        cache_key = (id(quarterly), id(avail_frame))
        cached = self._promote_cache.get(cache_key)
        if cached is not None and cached[0] is quarterly and cached[1] is avail_frame:
            self._promote_cache.move_to_end(cache_key)
            return cached[2]

        cal = self.trading_calendar
        avail_aligned = avail_frame.reindex(index=quarterly.index, columns=quarterly.columns)
        out = pd.DataFrame(np.nan, index=cal, columns=quarterly.columns, dtype="float64")
        cal_values = cal.to_numpy(dtype="datetime64[ns]")

        for ticker in quarterly.columns:
            values = quarterly[ticker].to_numpy(dtype="float64")
            keys = pd.to_datetime(avail_aligned[ticker]).to_numpy(dtype="datetime64[ns]")

            valid = ~pd.isna(keys)
            if not valid.any():
                continue
            keys, values = keys[valid], values[valid]

            order = np.argsort(keys, kind="stable")
            keys, values = keys[order], values[order]

            # searchsorted(side="right") - 1  ⇒  datekey <= d 인 마지막 인덱스
            pos = np.searchsorted(keys, cal_values, side="right") - 1
            picked = np.where(pos >= 0, values[np.clip(pos, 0, None)], np.nan)
            out[ticker] = picked

        self._promote_cache[cache_key] = (quarterly, avail_frame, out)
        while len(self._promote_cache) > PROMOTE_CACHE_SIZE:
            self._promote_cache.popitem(last=False)
        return out

    def eval_daily(self, expr: object) -> pd.DataFrame:
        """
        표현식을 평가해 일별 그리드 프레임으로 돌려준다.

        분기 그리드 결과는 자신의 공시일(avail)로 승격된다 — 파이프라인과
        유니버스 필터가 쓰는 표준 진입점.
        """
        panel = expr.eval(self)  # type: ignore[attr-defined]
        if panel.grid == "daily":
            return panel.data
        return self.to_daily(panel.data, panel.avail)

    @property
    def trading_calendar(self) -> pd.DatetimeIndex:
        if self.calendar is not None:
            return self.calendar
        for frame in self.daily.values():
            return pd.DatetimeIndex(frame.index)
        raise MissingDataError("평가 캘린더를 결정할 수 없습니다 (daily 데이터 없음).")

    # ------------------------------------------------------- 그룹 / 중립화 지원
    def groups(self, key: str, panel: Panel) -> pd.DataFrame:
        """그룹 라벨을 팩터 패널과 같은 모양으로 펼친다."""
        if key not in self.meta:
            raise MissingDataError(f"그룹 키 '{key}' 이(가) meta 에 없습니다.")
        labels = self.meta[key].reindex(panel.data.columns)
        return pd.DataFrame(
            np.tile(labels.to_numpy(), (len(panel.data.index), 1)),
            index=panel.data.index,
            columns=panel.data.columns,
        )

    def design_matrix(self, keys: tuple[str, ...], panel: Panel) -> dict[str, pd.DataFrame]:
        """
        중립화용 설계행렬을 만든다.

        - "sector"/"industry" 등 범주형 → 원-핫 더미 (기준 카테고리 1개 제외)
        - "size" → 로그 시가총액 (연속 통제변수)
        """
        design: dict[str, pd.DataFrame] = {}
        index, columns = panel.data.index, panel.data.columns

        for key in keys:
            if key == "size":
                mcap = self._aligned_daily("mcap", index, columns)
                design["size"] = np.log(mcap.where(mcap > 0))
                continue
            if key == "beta":
                design["beta"] = self._aligned_daily("beta", index, columns)
                continue

            labels = self.meta.get(key)
            if labels is None:
                raise MissingDataError(f"중립화 키 '{key}' 이(가) meta 에 없습니다.")
            labels = labels.reindex(columns)
            categories = sorted(labels.dropna().unique())[1:]  # 더미 트랩 회피
            for cat in categories:
                dummy = labels.eq(cat).astype(float).where(labels.notna())
                design[f"{key}={cat}"] = pd.DataFrame(
                    np.tile(dummy.to_numpy(dtype="float64"), (len(index), 1)),
                    index=index,
                    columns=columns,
                )
        return design

    # ------------------------------------------------------------------ 내부
    def _aligned_daily(self, name: str, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
        if name not in self.daily:
            raise MissingDataError(f"통제변수 '{name}' 이(가) daily 에 없습니다.")
        return self.daily[name].reindex(index=index, columns=columns)

    def _broadcast_meta(self, series: pd.Series) -> pd.DataFrame:
        cal = self.trading_calendar
        return pd.DataFrame(
            np.tile(series.to_numpy(), (len(cal), 1)), index=cal, columns=series.index
        )


class MissingDataError(KeyError):
    """컨텍스트에 필요한 데이터가 없을 때. 조용한 NaN 대신 명시적 실패."""
