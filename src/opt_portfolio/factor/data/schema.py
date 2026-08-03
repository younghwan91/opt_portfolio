"""
정규화 데이터 스키마

벤더(Sharadar/FMP/Polygon)마다 필드명이 다르므로, 팩터 정의가 벤더에 묶이지
않도록 표준 필드명을 정의하고 어댑터가 여기로 매핑한다.

퀀트 관점:
- `kind` (stock/flow) 구분이 TTM 계산의 정확성을 좌우한다.
  플로우(매출·순이익)는 4분기 합, 스톡(자산·자본)은 최신값.
- `grid` 는 표현식 평가 그리드를 결정하며, 분기/일별 혼합 연산 시
  PIT 승격 경로를 강제하는 근거가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldKind = Literal["stock", "flow", "price", "ratio", "meta"]
FieldGrid = Literal["quarterly", "daily"]
SourceTable = Literal["SF1", "SEP", "SF2", "SF3", "TICKERS", "ACTIONS", "FMP", "MACRO"]


@dataclass(frozen=True)
class FieldSpec:
    """표준 필드 하나의 메타데이터."""

    name: str
    kind: FieldKind
    grid: FieldGrid
    source: SourceTable
    vendor: dict[str, str]
    desc: str

    @property
    def is_flow(self) -> bool:
        return self.kind == "flow"


def _f(
    name: str,
    kind: FieldKind,
    grid: FieldGrid,
    source: SourceTable,
    desc: str,
    *,
    sharadar: str | None = None,
    fmp: str | None = None,
) -> FieldSpec:
    vendor: dict[str, str] = {}
    if sharadar:
        vendor["sharadar"] = sharadar
    if fmp:
        vendor["fmp"] = fmp
    return FieldSpec(name=name, kind=kind, grid=grid, source=source, vendor=vendor, desc=desc)


# --------------------------------------------------------------- 손익계산서 (flow)
_INCOME = [
    _f("revenue", "flow", "quarterly", "SF1", "매출액", sharadar="revenue", fmp="revenue"),
    _f("cor", "flow", "quarterly", "SF1", "매출원가", sharadar="cor", fmp="costOfRevenue"),
    _f("gp", "flow", "quarterly", "SF1", "매출총이익", sharadar="gp", fmp="grossProfit"),
    _f("opex", "flow", "quarterly", "SF1", "영업비용", sharadar="opex", fmp="operatingExpenses"),
    _f("sgna", "flow", "quarterly", "SF1", "판매관리비", sharadar="sgna",
       fmp="sellingGeneralAndAdministrativeExpenses"),
    _f("rnd", "flow", "quarterly", "SF1", "연구개발비", sharadar="rnd",
       fmp="researchAndDevelopmentExpenses"),
    _f("opinc", "flow", "quarterly", "SF1", "영업이익", sharadar="opinc", fmp="operatingIncome"),
    _f("ebit", "flow", "quarterly", "SF1", "EBIT", sharadar="ebit", fmp="operatingIncome"),
    _f("ebitda", "flow", "quarterly", "SF1", "EBITDA", sharadar="ebitda", fmp="ebitda"),
    _f("intexp", "flow", "quarterly", "SF1", "이자비용", sharadar="intexp", fmp="interestExpense"),
    _f("taxexp", "flow", "quarterly", "SF1", "법인세비용", sharadar="taxexp",
       fmp="incomeTaxExpense"),
    _f("netinc", "flow", "quarterly", "SF1", "당기순이익", sharadar="netinc", fmp="netIncome"),
    _f("epsdil", "flow", "quarterly", "SF1", "희석주당순이익", sharadar="epsdil", fmp="epsdiluted"),
    _f("depamor", "flow", "quarterly", "SF1", "감가상각비", sharadar="depamor",
       fmp="depreciationAndAmortization"),
    _f("sbcomp", "flow", "quarterly", "SF1", "주식보상비용", sharadar="sbcomp",
       fmp="stockBasedCompensation"),
]

# ------------------------------------------------------------- 재무상태표 (stock)
_BALANCE = [
    _f("assets", "stock", "quarterly", "SF1", "총자산", sharadar="assets", fmp="totalAssets"),
    _f("assetsc", "stock", "quarterly", "SF1", "유동자산", sharadar="assetsc",
       fmp="totalCurrentAssets"),
    _f("assetsnc", "stock", "quarterly", "SF1", "비유동자산", sharadar="assetsnc",
       fmp="totalNonCurrentAssets"),
    _f("liabilities", "stock", "quarterly", "SF1", "총부채", sharadar="liabilities",
       fmp="totalLiabilities"),
    _f("liabilitiesc", "stock", "quarterly", "SF1", "유동부채", sharadar="liabilitiesc",
       fmp="totalCurrentLiabilities"),
    _f("equity", "stock", "quarterly", "SF1", "자기자본", sharadar="equity",
       fmp="totalStockholdersEquity"),
    _f("debt", "stock", "quarterly", "SF1", "총차입금", sharadar="debtusd", fmp="totalDebt"),
    _f("debtc", "stock", "quarterly", "SF1", "단기차입금", sharadar="debtc", fmp="shortTermDebt"),
    _f("debtnc", "stock", "quarterly", "SF1", "장기차입금", sharadar="debtnc", fmp="longTermDebt"),
    _f("cashneq", "stock", "quarterly", "SF1", "현금성자산", sharadar="cashnequsd",
       fmp="cashAndCashEquivalents"),
    _f("inventory", "stock", "quarterly", "SF1", "재고자산", sharadar="inventory",
       fmp="inventory"),
    _f("receivables", "stock", "quarterly", "SF1", "매출채권", sharadar="receivables",
       fmp="netReceivables"),
    _f("intangibles", "stock", "quarterly", "SF1", "무형자산", sharadar="intangibles",
       fmp="goodwillAndIntangibleAssets"),
    _f("tangibles", "stock", "quarterly", "SF1", "유형자산", sharadar="tangibles"),
    _f("invcap", "stock", "quarterly", "SF1", "투하자본", sharadar="invcap"),
    _f("invcapavg", "stock", "quarterly", "SF1", "평균 투하자본", sharadar="invcapavg"),
    _f("assetsavg", "stock", "quarterly", "SF1", "평균 총자산", sharadar="assetsavg"),
    _f("equityavg", "stock", "quarterly", "SF1", "평균 자기자본", sharadar="equityavg"),
    _f("retearn", "stock", "quarterly", "SF1", "이익잉여금", sharadar="retearn",
       fmp="retainedEarnings"),
    _f("workingcapital", "stock", "quarterly", "SF1", "운전자본", sharadar="workingcapital"),
    _f("sharesbas", "stock", "quarterly", "SF1", "발행주식수", sharadar="sharesbas"),
    _f("shareswadil", "stock", "quarterly", "SF1", "희석 가중평균주식수", sharadar="shareswadil"),
]

# ------------------------------------------------------------- 현금흐름표 (flow)
_CASHFLOW = [
    _f("ncfo", "flow", "quarterly", "SF1", "영업활동현금흐름", sharadar="ncfo",
       fmp="netCashProvidedByOperatingActivities"),
    _f("capex", "flow", "quarterly", "SF1", "자본적지출", sharadar="capex",
       fmp="capitalExpenditure"),
    _f("fcf", "flow", "quarterly", "SF1", "잉여현금흐름", sharadar="fcf", fmp="freeCashFlow"),
    _f("ncfdiv", "flow", "quarterly", "SF1", "배당지급 (유출이라 음수)", sharadar="ncfdiv",
       fmp="dividendsPaid"),
    _f("ncfcommon", "flow", "quarterly", "SF1", "자사주매입 순액 (유출이라 음수)",
       sharadar="ncfcommon", fmp="commonStockRepurchased"),
    _f("ncfdebt", "flow", "quarterly", "SF1", "차입금 순증감", sharadar="ncfdebt"),
    _f("dps", "flow", "quarterly", "SF1", "주당배당금", sharadar="dps"),
]

# ---------------------------------------------------------------- 가격 (daily)
_PRICE = [
    _f("close", "price", "daily", "SEP", "수정 종가", sharadar="closeadj", fmp="adjClose"),
    _f("closeunadj", "price", "daily", "SEP", "수정 전 종가", sharadar="closeunadj", fmp="close"),
    _f("open", "price", "daily", "SEP", "시가", sharadar="open", fmp="open"),
    _f("high", "price", "daily", "SEP", "고가", sharadar="high", fmp="high"),
    _f("low", "price", "daily", "SEP", "저가", sharadar="low", fmp="low"),
    _f("volume", "price", "daily", "SEP", "거래량", sharadar="volume", fmp="volume"),
    _f("mcap", "price", "daily", "SEP", "시가총액 (일별)", sharadar="marketcap",
       fmp="marketCap"),
    _f("ev", "price", "daily", "SEP", "기업가치", sharadar="ev", fmp="enterpriseValue"),
    _f("dividends", "price", "daily", "SEP", "배당락 배당금", sharadar="dividends"),
]

# ------------------------------------------------------------- 수급 프록시 / 기타
_FLOW_PROXY = [
    _f("inst_shares", "stock", "quarterly", "SF3", "13F 기관 보유 주식수 합계"),
    _f("inst_holders", "stock", "quarterly", "SF3", "13F 보고 기관 수"),
    _f("insider_net_shares", "flow", "quarterly", "SF2", "내부자 순매수 주식수 (Form 4)"),
    _f("short_interest", "stock", "daily", "FMP", "공매도 잔고 주식수"),
]

# ------------------------------------------------- 애널리스트 추정치 (FMP 전용)
_ESTIMATES = [
    _f("eps_growth_fwd", "ratio", "quarterly", "FMP", "선행 EPS 성장률 (컨센서스)",
       fmp="estimatedEpsGrowth"),
    _f("eps_est_fwd", "ratio", "quarterly", "FMP", "선행 EPS 컨센서스", fmp="estimatedEpsAvg"),
]

_META = [
    _f("sector", "meta", "daily", "TICKERS", "GICS 섹터", sharadar="sector"),
    _f("industry", "meta", "daily", "TICKERS", "GICS 산업", sharadar="industry"),
    _f("siccode", "meta", "daily", "TICKERS", "SIC 코드", sharadar="siccode"),
    _f("location", "meta", "daily", "TICKERS", "본사 소재지 (중국기업 필터용)",
       sharadar="location"),
    _f("category", "meta", "daily", "TICKERS", "증권 유형 (ADR/PTP 식별)", sharadar="category"),
    _f("is_delisted", "meta", "daily", "TICKERS", "상장폐지 여부", sharadar="isdelisted"),
]

FIELDS: dict[str, FieldSpec] = {
    spec.name: spec
    for spec in (
        *_INCOME, *_BALANCE, *_CASHFLOW, *_PRICE, *_FLOW_PROXY, *_ESTIMATES, *_META
    )
}


def get_field(name: str) -> FieldSpec:
    try:
        return FIELDS[name]
    except KeyError:
        raise UnknownFieldError(
            f"알 수 없는 필드 '{name}'. 사용 가능한 필드는 schema.FIELDS 참조."
        ) from None


def fields_by_source(source: SourceTable) -> list[FieldSpec]:
    return [spec for spec in FIELDS.values() if spec.source == source]


class UnknownFieldError(KeyError):
    """스키마에 없는 필드를 참조했을 때."""


# ------------------------------------------------------ PIT 규약 (모든 어댑터 공통)

#: 재무 데이터 조회 시 반드시 이 컬럼으로 필터링한다. 회계기간말(calendardate)로
#: 조인하면 결산일과 공시일 사이 최대 90일의 미래 정보가 새어 들어간다.
PIT_DATE_COLUMN = "datekey"

#: 회계기간말. 정렬·성장률 계산의 순서 기준으로만 쓰고, 가용성 판단에는 쓰지 않는다.
PERIOD_DATE_COLUMN = "calendardate"

#: as-reported 분기. 재작성(restated) 값을 쓰면 look-ahead 가 발생한다.
DEFAULT_DIMENSION = "ARQ"

#: 13F 는 분기말 + 45일 공시. 프로바이더가 공시일을 주지 않으면 이 값으로 보정한다.
FILING_LAG_13F_DAYS = 45
