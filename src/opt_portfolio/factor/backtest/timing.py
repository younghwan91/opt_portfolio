"""
마켓 타이밍 오버레이 — 모멘텀 / 재진입 히스테리시스 / 매크로

퀀트 관점:
- 200일 이평 타이밍의 가치는 수익률 향상이 아니라 **좌측 꼬리 절단**이다
  (Faber 2007). 기대수익은 비슷하거나 낮아지고 MDD 가 줄어든다.
- 이평선 근처에서 매일 사고파는 휩쏘(whipsaw)가 이 전략의 실질 비용이다.
  재진입 히스테리시스(연속 k 일 상회 시에만 복귀)가 그 방어책.
- 상태 의존 로직이라 벡터화하지 않고 명시적 상태기계로 쓴다 —
  6,000일 루프는 순수 파이썬으로도 밀리초 단위다.
"""

from __future__ import annotations

import pandas as pd


def momentum_exposure(
    benchmark_close: pd.Series,
    ma_days: int = 200,
    reentry_days: int = 0,
) -> pd.Series:
    """
    지수 vs 이동평균 기반 익스포저 (0.0 = 전량 현금, 1.0 = 전량 투자).

    Args:
        benchmark_close: 벤치마크 (S&P500 등) 종가
        ma_days: 이동평균 기간
        reentry_days: 청산 후 재진입에 필요한 연속 상회 일수.
            0 이면 히스테리시스 없이 즉시 재진입 (순수 Faber 규칙).

    신호는 t 종가로 판정하고 **t+1 부터 적용해야 한다** — 시프트는
    엔진이 담당하므로 여기서는 판정일 기준 그대로 반환한다.
    """
    ma = benchmark_close.rolling(ma_days, min_periods=ma_days).mean()
    above = benchmark_close > ma

    if reentry_days <= 0:
        return above.astype(float).where(ma.notna(), 1.0)

    exposure = pd.Series(1.0, index=benchmark_close.index)
    invested = True
    streak = 0
    for date in benchmark_close.index:
        if pd.isna(ma.loc[date]):
            exposure.loc[date] = 1.0  # 워밍업 구간은 투자 상태로 시작
            continue
        if invested:
            if not above.loc[date]:
                invested, streak = False, 0
        else:
            streak = streak + 1 if above.loc[date] else 0
            if streak >= reentry_days:
                invested = True
        exposure.loc[date] = 1.0 if invested else 0.0
    return exposure


def macro_exposure(
    signals: dict[str, pd.Series],
    reduction: float = 0.5,
) -> pd.Series:
    """
    매크로 리스크오프 신호의 합성 익스포저.

    Args:
        signals: {신호명: 불리언 시리즈 (True = 위험 신호)}.
            예: 장단기금리차 역전, 실업률 12개월 이평 상회.
        reduction: 위험 신호 1개당 익스포저 축소 배율.
            신호 k개 발동 시 exposure = reduction^k.

    데이터 어댑터가 붙기 전에도 인터페이스를 고정해 두는 목적 —
    신호 시리즈는 반드시 **발표일 기준**으로 정렬되어 있어야 한다
    (실업률 통계는 익월 발표: 발표 전 값을 쓰면 look-ahead).
    """
    if not signals:
        raise ValueError("매크로 신호가 없습니다")
    aligned = pd.concat(signals.values(), axis=1).fillna(False)
    n_active = aligned.sum(axis=1)
    return pd.Series(reduction, index=aligned.index) ** n_active


def combine_exposures(*exposures: pd.Series) -> pd.Series:
    """복수 오버레이는 곱으로 결합 — 가장 보수적인 신호가 지배한다."""
    combined = exposures[0]
    for e in exposures[1:]:
        combined, e = combined.align(e, join="outer")
        combined = combined.fillna(1.0) * e.fillna(1.0)
    return combined.clip(0.0, 1.0)


def volatility_target_exposure(
    returns: pd.Series,
    target_vol: float = 0.15,
    *,
    window: int = 63,
    max_exposure: float = 1.0,
    min_exposure: float = 0.0,
    ann: int = 252,
) -> pd.Series:
    """
    변동성 타게팅 익스포저 — 이진 타이밍의 연속판.

        exposure_t = clip(target_vol / realized_vol_t, min, max)

    실현변동성이 목표보다 높으면 줄이고 낮으면 늘린다. Moreira & Muir(2017)의
    변동성 관리 포트폴리오와 같은 형태다. 이진 오버레이가 못 하는 두 가지를
    한다 — 변동성이 서서히 오를 때 점진적으로 줄이고, 조용한 구간에서
    노출을 회복한다.

    Args:
        returns: 벤치마크 또는 전략의 일별 수익률
        window: 실현변동성 측정 창 (거래일)
        max_exposure: 상한. 기본 1.0 = 레버리지 없음. 변동성이 0 에 가까우면
            비율이 발산하므로 상한이 없으면 위험하다.
        min_exposure: 하한. 0 이면 완전 청산을 허용한다. 이진 타이밍이
            2009년 반등 초입을 놓친 것이 완전 청산의 대가였다.

    관측이 모자란 초기 구간은 **1.0(중립)** 이다 — 0 을 주면 그 기간이
    통째로 사라진다. 신호는 판정일 기준이며 시프트는 엔진이 담당한다.
    """
    realized = returns.rolling(window, min_periods=window).std(ddof=1) * (ann**0.5)
    exposure = (target_vol / realized.where(realized > 0)).clip(min_exposure, max_exposure)
    return exposure.fillna(1.0)
