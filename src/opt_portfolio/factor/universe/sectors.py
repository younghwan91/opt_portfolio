"""
섹터 분류 — Sharadar 섹터 체계 + WICS(한국) 매핑

Sharadar TICKERS 테이블은 Morningstar 스타일 11개 섹터를 쓴다.
요청된 26개 WICS 산업 분류를 이 체계로 매핑해, 한국 플랫폼의
사고방식 그대로 유니버스를 정의할 수 있게 한다.

주의: WICS 는 한국 시장 구조를 반영한 분류라 1:1 대응이 없는 항목이 있다
(조선·상사 등). 매핑은 (섹터, 산업 키워드) 쌍이며, 키워드가 있으면
industry 문자열 부분일치로 좁힌다. 정밀 분류가 필요하면 siccode 를 쓸 것.
"""

from __future__ import annotations

import pandas as pd

#: Sharadar/Morningstar 섹터 전체
SECTORS = (
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
)

#: '금융주 제외' 필터 대상
FINANCIAL_SECTORS = frozenset({"Financial Services", "Real Estate"})

#: WICS 산업명 → (섹터, industry 부분일치 키워드 튜플)
#: 키워드 () 는 섹터 전체를 의미한다.
WICS_TO_SECTOR: dict[str, tuple[str, tuple[str, ...]]] = {
    "건강관리": ("Healthcare", ()),
    "자동차": ("Consumer Cyclical", ("Auto",)),
    "화장품,의류,완구": ("Consumer Cyclical", ("Apparel", "Personal", "Leisure")),
    "보험": ("Financial Services", ("Insurance",)),
    "필수소비재": ("Consumer Defensive", ()),
    "운송": ("Industrials", ("Airlines", "Railroads", "Trucking", "Marine", "Logistics")),
    "상사,자본재": ("Industrials", ("Conglomerates", "Capital", "Distribution")),
    "비철,목재등": ("Basic Materials", ("Aluminum", "Copper", "Lumber", "Paper", "Metals")),
    "화학": ("Basic Materials", ("Chemicals",)),
    "건설,건축관련": ("Industrials", ("Construction", "Building", "Engineering")),
    "에너지": ("Energy", ()),
    "기계": ("Industrials", ("Machinery", "Tools")),
    "철강": ("Basic Materials", ("Steel",)),
    "반도체": ("Technology", ("Semiconductor",)),
    "IT하드웨어": ("Technology", ("Hardware", "Computer", "Electronic")),
    "통신서비스": ("Communication Services", ("Telecom",)),
    "증권": ("Financial Services", ("Capital Markets", "Brokers", "Asset Management")),
    "디스플레이": ("Technology", ("Display", "Optical")),
    "IT가전": ("Technology", ("Consumer Electronics",)),
    "소매(유통)": ("Consumer Cyclical", ("Retail", "Department", "Specialty")),
    "유틸리티": ("Utilities", ()),
    "미디어,교육": ("Communication Services", ("Media", "Entertainment", "Education")),
    "은행": ("Financial Services", ("Banks",)),
    "호텔,레저서비스": ("Consumer Cyclical", ("Lodging", "Resorts", "Restaurants", "Gambling")),
    "소프트웨어": ("Technology", ("Software", "Information Technology")),
    "조선": ("Industrials", ("Marine Shipping", "Aerospace")),  # 미장 근사 — 순수 조선업 부재
}


def wics_mask(
    sector: pd.Series,
    industry: pd.Series,
    selected: list[str],
) -> pd.Series:
    """
    선택된 WICS 산업명 목록 → 티커 불리언 마스크.

    Args:
        sector / industry: (ticker → 라벨) 시리즈 (PanelContext.meta)
        selected: WICS 산업명. 전체 선택이면 이 함수를 부르지 말 것.
    """
    unknown = [w for w in selected if w not in WICS_TO_SECTOR]
    if unknown:
        raise KeyError(f"알 수 없는 WICS 산업명: {unknown}")

    mask = pd.Series(False, index=sector.index)
    for wics in selected:
        target_sector, keywords = WICS_TO_SECTOR[wics]
        in_sector = sector.eq(target_sector)
        if not keywords:
            mask |= in_sector
            continue
        pattern = "|".join(keywords)
        in_industry = industry.fillna("").str.contains(pattern, case=False, regex=True)
        mask |= in_sector & in_industry
    return mask
