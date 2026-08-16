"""사전 등록 9개 구성 — 이 목록이 곧 DSR 의 `n_trials` 다.

**늘리지 않는다.** 늘리면 DSR 에 정산되지 않는 탐색이 되고, 그것이 이 저장소가
이미 지고 있는 부채다 (`07-experiment-log.md` §6 — 124개를 훑어 고른 행위 자체가
정산되지 않았다).

개선안을 둘로 제한한 근거:
- 200일 이평 — 팩터 엔진에서 MDD −63.8% → −23.7%. 이 저장소 최대의 실측 개선
- 리밸런싱 트랜치 — timing luck 은 문서화된 약점이고, 분산을 줄이는 장치라
  수익을 좇는 파라미터가 아니다

카나리아 구성 변경·모멘텀 가중치 조정·보유 수 탐색은 근거가 없어 뺐다.
"""

from __future__ import annotations

from .strategy import StrategySpec

_BAA_CANARY = ("SPY", "EFA", "EEM", "AGG")
_BAA_DEFENSIVE = ("TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "AGG")
_BAA_BAL_OFFENSIVE = (
    "SPY",
    "QQQ",
    "IWM",
    "VGK",
    "EWJ",
    "EEM",
    "VNQ",
    "DBC",
    "GLD",
    "TLT",
    "HYG",
    "LQD",
)


def _spy() -> StrategySpec:
    return StrategySpec(
        name="spy",
        canary=(),
        offensive=(),
        defensive=(),
        top_n_offensive=0,
        top_n_defensive=0,
        static_weights={"SPY": 1.0},
    )


def _static_60_40() -> StrategySpec:
    return StrategySpec(
        name="static_60_40",
        canary=(),
        offensive=(),
        defensive=(),
        top_n_offensive=0,
        top_n_defensive=0,
        static_weights={"SPY": 0.6, "IEF": 0.4},
    )


def _vaa_g4() -> StrategySpec:
    # 경보기와 투자 대상이 같다 — 이것이 VAA 의 병이다.
    return StrategySpec(
        name="vaa_g4",
        canary=("SPY", "EFA", "EEM", "AGG"),
        offensive=("SPY", "EFA", "EEM", "AGG"),
        defensive=("LQD", "IEF", "SHY"),
        top_n_offensive=1,
        top_n_defensive=1,
        selection="13612w",
        cash_ticker=None,
    )


def _baa(name: str, offensive: tuple[str, ...], top_off: int, top_def: int) -> StrategySpec:
    return StrategySpec(
        name=name,
        canary=_BAA_CANARY,
        offensive=offensive,
        defensive=_BAA_DEFENSIVE,
        top_n_offensive=top_off,
        top_n_defensive=top_def,
        selection="sma13",
        cash_ticker="BIL",
    )


_BAA_AGG = _baa("baa_agg", ("QQQ", "EEM", "EFA", "AGG"), 1, 3)
_BAA_BAL = _baa("baa_bal", _BAA_BAL_OFFENSIVE, 6, 3)

REGISTERED: dict[str, StrategySpec] = {
    "spy": _spy(),
    "static_60_40": _static_60_40(),
    "vaa_g4": _vaa_g4(),
    "baa_agg": _BAA_AGG,
    "baa_bal": _BAA_BAL,
    # 변형은 같은 스펙을 쓰고 실행 단계에서 오버레이·트랜치를 얹는다.
    "baa_agg_ma": _baa("baa_agg_ma", ("QQQ", "EEM", "EFA", "AGG"), 1, 3),
    "baa_bal_ma": _baa("baa_bal_ma", _BAA_BAL_OFFENSIVE, 6, 3),
    "baa_bal_tranche": _baa("baa_bal_tranche", _BAA_BAL_OFFENSIVE, 6, 3),
    "baa_bal_ma_tranche": _baa("baa_bal_ma_tranche", _BAA_BAL_OFFENSIVE, 6, 3),
}

#: 200일 이평 오버레이를 적용할 구성
MA_OVERLAY: frozenset[str] = frozenset({"baa_agg_ma", "baa_bal_ma", "baa_bal_ma_tranche"})

#: 리밸런싱 트랜치를 적용할 구성
TRANCHE: frozenset[str] = frozenset({"baa_bal_tranche", "baa_bal_ma_tranche"})

#: DSR 에 넘길 시도 횟수. 이 저장소의 주요 구조적 약점은 탐색 횟수 미정산
#: (`07-experiment-log.md` §6). 이 목록을 늘리면 정산되지 않은 탐색이 되므로,
#: 이 값은 `len(REGISTERED)` 로 유도되고 수동 수정으로부터 보호된다.
N_TRIALS: int = len(REGISTERED)
