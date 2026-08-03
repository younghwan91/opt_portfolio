"""
가속 팩터 (15개)

- 재무 성장 가속 12개: derive_acceleration() 으로 자동 생성 (6 × 2)
- 이동평균 모멘텀 가속 3개: 파라미터화된 헬퍼 하나로 생성

퀀트 관점:
- 가속 = 성장률의 차분 = 2차 미분. 노이즈가 두 번 증폭되므로 단독 사용은
  위험하다. 윈저라이즈를 2%로 강하게 걸고, IC 검증에서 대부분 탈락할 것을
  전제로 둔다 — 탈락을 확인하는 것이 검증 레이어의 역할이다.
"""

from __future__ import annotations

from opt_portfolio.factor.dsl.expr import Expr, F
from opt_portfolio.factor.dsl.registry import derive_acceleration, factor

# --------------------------------------------------- 재무 성장 가속 (12개, 자동 생성)

_ACCEL_BASES = [
    (F.netinc, "NETINC", "순이익"),
    (F.revenue, "REVENUE", "매출액"),
    (F.gp, "GP", "매출총이익"),
    (F.opinc, "OPINC", "영업이익"),
    (F.ncfo, "NCFO", "영업활동현금흐름"),
    (F.netinc - F.ncfo, "ACCRUAL", "발생액"),
]

ACCEL_FACTORS = {
    stem: derive_acceleration(expr, stem, label) for expr, stem, label in _ACCEL_BASES
}


# ----------------------------------------------- 이동평균 모멘텀 가속 (3개)


def ma_momentum_accel(short: int, long: int, lookback: int) -> Expr:
    """
    이동평균 비율의 변화량.

    `MA_short / MA_long` 이 추세의 강도라면, 그 lookback 일 변화량은
    추세가 강해지는 중인지 약해지는 중인지를 나타낸다.

    Args:
        short: 단기 이평 기간 (개월 → 21영업일 환산)
        long: 장기 이평 기간 (개월 → 21영업일 환산)
        lookback: 변화량 관측 기간 (영업일)

    Note:
        (a/b/c) 파라미터 해석은 (단기/장기/관측) 으로 **추정**한 것이다.
        실제 정의가 다르면 이 함수 하나만 고치면 세 팩터가 모두 따라온다.
    """
    ratio = F.close.ma(short * 21) / F.close.ma(long * 21)
    return ratio - ratio.lag(lookback)


_MA_ACCEL_PARAMS = [(3, 3, 10), (3, 12, 5), (10, 1, 5)]

MA_ACCEL_FACTORS = {
    (s, lng, k): factor(
        f"MA_ACCEL_{s}_{lng}_{k}",
        ma_momentum_accel(s, lng, k),
        category="acceleration",
        label=f"이동평균 모멘텀 가속 ({s}/{lng}/{k})",
        winsor=0.02,
        requires=("SEP",),
        notes="[파라미터 해석 추정] (단기개월/장기개월/관측일)",
    )
    for s, lng, k in _MA_ACCEL_PARAMS
}

__all__ = ["ACCEL_FACTORS", "MA_ACCEL_FACTORS", "ma_momentum_accel"]
