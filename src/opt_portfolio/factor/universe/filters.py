"""
유니버스 필터 — 요청된 퀀터스식 필터의 미장 구현

퀀트 관점:
- 유동성·페니스톡 필터는 알파가 아니라 **실현 가능성**이다. 이걸 끄면
  백테스트는 체결 불가능한 종목에서 수익을 만든다.
- 적자기업 필터는 분기 데이터의 **공시일 기준**으로 적용해야 한다.
  회계기간말 기준으로 걸면 실적 발표 전에 적자를 '미리 알고' 빼는
  look-ahead 가 된다 — ctx.eval_daily() 가 이를 보장한다.
- PTP 필터는 백테스트 성과에 안 잡히는 실비용(매도액 10% 원천징수)
  방어다. 종목 식별은 IRS 공식 리스트가 정답이며, 여기의 이름 패턴 +
  시드 리스트는 휴리스틱이다 — 실전 전에 브로커 리스트로 검증할 것.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from opt_portfolio.factor.dsl.context import MissingDataError, PanelContext
from opt_portfolio.factor.dsl.expr import F
from opt_portfolio.factor.library._composites import ALTMAN_DISTRESS, altman_z_expr
from opt_portfolio.factor.universe.sectors import FINANCIAL_SECTORS, wics_mask

logger = logging.getLogger(__name__)

#: 대표적 PTP (IRS Sec.1446(f) 대상) 시드 — 휴리스틱 보조용, 완전 목록 아님
PTP_SEED = frozenset(
    {
        "ET",
        "EPD",
        "MPLX",
        "PAA",
        "WES",
        "AM",
        "CQP",
        "ARLP",
        "SUN",
        "NS",
        "GLP",
        "USAC",
        "DKL",
        "HESM",
        "MMP",
        "BSM",
        "CAPL",
        "GEL",
        "NGL",
    }
)

#: 지주사/PTP 이름 패턴 (비캡처 그룹 — str.contains 경고 방지)
_PTP_NAME = r"\b(?:L\.?P\.?|Partners)\b"
_HOLDING_NAME = r"\bHold(?:ing|ings)\b"


@dataclass(frozen=True)
class UniverseConfig:
    """유니버스 정의 — 전부 요청된 UI 항목의 미장 대응."""

    # 일반 필터
    exclude_financials: bool = True
    exclude_holdings: bool = False  # 미장 의미 약함 — 기본 off
    exclude_distressed: bool = True  # '관리종목 제외' 대체: Altman Z < 1.81
    exclude_deficit_quarter: bool = False  # 최근 분기 순이익 ≤ 0
    exclude_deficit_ttm: bool = False  # TTM 순이익 ≤ 0
    exclude_china: bool = True
    exclude_ptp: bool = True
    extra_ptp_tickers: tuple[str, ...] = ()
    smallcap_bottom_pct: float | None = None  # 0.2 → 시총 하위 20% 만

    # 실현 가능성 필터 (요청에 없지만 기본 on — 끄려면 명시적으로)
    min_price_usd: float = 5.0  # 수정 전 종가 기준
    min_adv_usd: float = 1_000_000.0  # 20일 평균 거래대금

    # 산업 필터 — WICS 이름 또는 빈 튜플(전체)
    wics_industries: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()  # Sharadar 섹터명 직접 지정 (WICS 대신)

    def __post_init__(self) -> None:
        if self.wics_industries and self.sectors:
            raise ValueError("wics_industries 와 sectors 는 동시 지정 불가")


def build_universe(ctx: PanelContext, config: UniverseConfig) -> pd.DataFrame:
    """
    (date × ticker) 불리언 마스크.

    일별 필터는 그날 기준, 분기 필터는 공시일 기준으로 적용된다.
    """
    close = ctx.daily.get("close")
    if close is None:
        raise ValueError("유니버스 구성에 close 가 필요합니다")
    mask = close.notna()

    # ---- 실현 가능성 (일별)
    if config.min_price_usd > 0:
        price = ctx.daily.get("closeunadj", close)
        mask &= price >= config.min_price_usd

    if config.min_adv_usd > 0 and "volume" in ctx.daily:
        adv = (ctx.daily["volume"] * close).rolling(20, min_periods=5).mean()
        mask &= adv >= config.min_adv_usd

    if config.smallcap_bottom_pct is not None and "mcap" in ctx.daily:
        pct_rank = ctx.daily["mcap"].rank(axis=1, pct=True)
        mask &= pct_rank <= config.smallcap_bottom_pct

    # ---- 메타 기반 (티커 단위 → 브로드캐스트)
    static = pd.Series(True, index=close.columns)

    sector = _meta(ctx, "sector", close.columns)
    industry = _meta(ctx, "industry", close.columns)
    name = _meta(ctx, "name", close.columns).fillna("")
    location = _meta(ctx, "location", close.columns).fillna("")

    if config.exclude_financials:
        static &= ~sector.isin(FINANCIAL_SECTORS)

    if config.exclude_holdings:
        static &= ~name.str.contains(_HOLDING_NAME, case=False, regex=True)

    if config.exclude_china:
        static &= ~location.str.contains("China|Hong Kong", case=False, regex=True)

    if config.exclude_ptp:
        ptp = name.str.contains(_PTP_NAME, case=True, regex=True) | close.columns.isin(
            PTP_SEED | set(config.extra_ptp_tickers)
        ).astype(bool)
        static &= ~pd.Series(ptp, index=close.columns)

    if config.wics_industries:
        static &= wics_mask(sector, industry, list(config.wics_industries))
    elif config.sectors:
        static &= sector.isin(config.sectors)

    mask &= pd.DataFrame(
        [static.to_numpy()] * len(mask.index), index=mask.index, columns=mask.columns
    )

    # ---- 재무 기반 (분기 → 공시일 기준 일별 승격)
    if config.exclude_deficit_quarter:
        netinc_daily = ctx.eval_daily(F.netinc)
        mask &= netinc_daily.reindex_like(close) > 0

    if config.exclude_deficit_ttm:
        netinc_ttm = ctx.eval_daily(F.netinc.ttm())
        mask &= netinc_ttm.reindex_like(close) > 0

    if config.exclude_distressed:
        try:
            z = ctx.eval_daily(altman_z_expr()).reindex_like(close)
        except MissingDataError as exc:
            # 구성 필드(운전자본·유보이익 등)가 스토어에 없으면 필터를 건너뛴다
            # — 죽는 것보다 경고가 낫다. 단, 로그는 반드시 남긴다.
            logger.warning("Altman Z 필터 생략 (필드 부족): %s", exc)
        else:
            # Z 미계산(재무 미공시·금융주) 종목은 통과 — 확인된 부실만 제외
            mask &= ~(z < ALTMAN_DISTRESS)

    return mask


def _meta(ctx: PanelContext, key: str, columns: pd.Index) -> pd.Series:
    series = ctx.meta.get(key)
    if series is None:
        return pd.Series(pd.NA, index=columns, dtype="object")
    return series.reindex(columns)
