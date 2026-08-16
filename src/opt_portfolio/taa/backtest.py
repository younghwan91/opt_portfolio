"""월별 리밸런싱 엔진.

**신호는 t월말 종가로 정하고 수익은 t+1월에 얻는다.** 같은 달 수익을 쓰면
룩어헤드이고, 그건 이 저장소가 구조로 막기로 한 실패 유형이다.

비용은 **회전한 만큼만** 문다. 비중이 안 바뀌면 0 이다 — 매달 물리면 정적
배분 기준선이 부당하게 불리해져 비교가 망가진다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import momentum_13612w, sma_ratio, to_monthly
from .strategy import StrategySpec, is_defensive, select_weights


@dataclass(frozen=True)
class BacktestOutput:
    """백테스트 결과.

    `equity` 는 `returns` 보다 원소가 **하나 많다** — 맨 앞에 진입 시점의
    원금(10,000)이 붙는다. 그래야 `equity[-1] == equity[0] * (1+returns).prod()`
    가 항등식으로 성립한다 (첫 달 수익을 그 시점 자산가치에 두 번 복리로
    반영하지 않기 위함이다). **`len(equity)` 로 기간 수를 세거나 두 시리즈를
    위치로 zip 하면 하나씩 밀린다** — 기간 수는 항상 `len(returns)` 를 쓴다.
    """

    returns: pd.Series
    equity: pd.Series
    selections: pd.Series
    defensive_ratio: float


def _equity_from_returns(returns: pd.Series, entry_date: pd.Timestamp) -> pd.Series:
    """`run_backtest` 와 같은 규약으로 `equity` 를 만든다.

    진입 시점(`entry_date`)의 원금 10,000 을 맨 앞에 붙인다. 이걸 빼먹으면
    `len(equity) == len(returns)` 가 되어 `BacktestOutput` 의 문서화된 불변식이
    깨지고, `equity` 에서 계산하는 MDD 가 **첫 달 손실을 놓친다** — 이 저장소의
    채택 관문(MDD ≤ 20%)을 실제보다 낮게 보이게 만들 수 있다.
    """
    growth = (1 + returns).cumprod() * 10_000.0
    base = pd.Series([10_000.0], index=pd.DatetimeIndex([entry_date]))
    return pd.concat([base, growth])


def run_backtest(
    spec: StrategySpec,
    daily: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    cost_bps: float = 10.0,
) -> BacktestOutput:
    """월별 리밸런싱 백테스트.

    Args:
        spec: 전략 선언
        daily: 일별 배당조정 가격 패널
        start/end: 검증 구간 (None 이면 데이터 전체)
        cost_bps: 편도 거래비용 (bp). 회전한 비중에만 적용된다.
    """
    monthly = to_monthly(daily)
    mom = momentum_13612w(monthly)
    sel = sma_ratio(monthly, window=13)
    # fill_method=None — 기본값(pad)은 결측 월말가를 직전 값으로 메워 0% 수익으로
    # 둔갑시킨다. 결측은 NaN 으로 남겨야 아래에서 절단을 잡아낼 수 있다.
    fwd = monthly.pct_change(fill_method=None).shift(-1)

    needed = spec.tickers()
    usable = mom.dropna(how="any", subset=needed).index
    if start is not None:
        usable = usable[usable >= start]
    if end is not None:
        usable = usable[usable <= end]
    if len(usable) == 0:
        raise ValueError(f"[{spec.name}] 평가 가능한 시점이 없다 — 데이터 구간을 확인하라")

    month_index = monthly.index
    prev: dict[str, float] = {}
    dates, rets, picks, defensive_flags = [], [], [], []
    entry_date: pd.Timestamp | None = None  # 자본 곡선의 시작점 — 첫 회전 직전 시점

    for date in usable:
        pos = month_index.get_loc(date)
        if pos + 1 >= len(month_index):
            continue  # 다음 달이 없는 마지막 시점 — 수익을 매길 대상이 없다
        realization_date = month_index[pos + 1]

        nxt = fwd.loc[date]
        weights = select_weights(spec, mom, sel, date)
        missing = [t for t in weights if not np.isfinite(nxt[t])]
        if missing:
            # 마지막 달(다음 달이 없는 시점)은 위에서 이미 걸러졌다 — 여기 걸리는
            # NaN 은 구간 중간의 가격 공백이다. 조용히 건너뛰면 이 저장소가 반복해
            # 겪은 "성공 로그를 남긴 절단"이 된다. 예외를 던진다.
            raise ValueError(
                f"[{spec.name}] {realization_date.date()} 수익률 결측: {missing} — "
                "가격 데이터 공백을 의심하라 (조용한 절단 금지)"
            )
        gross = float(sum(w * nxt[t] for t, w in weights.items()))

        all_tickers = set(weights) | set(prev)
        turnover = sum(abs(weights.get(t, 0.0) - prev.get(t, 0.0)) for t in all_tickers)
        cost = turnover * cost_bps / 10_000.0

        if entry_date is None:
            entry_date = date
        dates.append(realization_date)
        rets.append(gross - cost)
        picks.append(",".join(sorted(weights)))
        defensive_flags.append(is_defensive(spec, mom, date))
        prev = weights

    returns = pd.Series(rets, index=pd.DatetimeIndex(dates), name=spec.name)
    if entry_date is not None:
        equity = _equity_from_returns(returns, entry_date)
    else:
        equity = (1 + returns).cumprod() * 10_000.0
    return BacktestOutput(
        returns=returns,
        equity=equity,
        selections=pd.Series(picks, index=returns.index),
        defensive_ratio=float(np.mean(defensive_flags)) if defensive_flags else 0.0,
    )


def run_with_ma_overlay(
    spec: StrategySpec,
    daily: pd.DataFrame,
    benchmark: str = "SPY",
    ma_days: int = 200,
    **kwargs: object,
) -> BacktestOutput:
    """벤치마크가 이평 아래면 그 달 수익을 0 으로 (현금).

    팩터 엔진에서 이 오버레이가 MDD 를 −63.8% → −23.7% 로 줄였다. 다만 BAA 의
    카나리아가 이미 추세를 판정하므로 **여기서는 효과가 없거나 마이너스일 수
    있다** — 이중 필터가 VAA 의 병(과도한 방어)을 재발시킬 수 있기 때문이다.
    설계 문서 §7 에 그 예상을 적어두었다.
    """
    base = run_backtest(spec, daily, **kwargs)  # type: ignore[arg-type]
    ma = daily[benchmark].rolling(ma_days, min_periods=ma_days).mean()
    # 불리언을 nullable "boolean" dtype 으로 유지해야 한다. 일반 bool 은 NaN 을
    # 못 담아 shift/reindex 가 만든 결측을 object dtype 으로 승격시키고, 뒤이은
    # fillna 가 그걸 다시 bool 로 내리며 FutureWarning(silent downcasting)을
    # 낸다 — Task 4 가 이 패키지 전체에서 없앤 경고를 되살리는 셈이라 여기서
    # 명시적으로 막는다.
    invested = (
        (daily[benchmark] > ma)
        .resample("ME")
        .last()
        .astype("boolean")
        .shift(1)
        .fillna(True)
        .astype(bool)
    )

    aligned = invested.reindex(base.returns.index).astype("boolean").fillna(True).astype(bool)
    returns = base.returns.where(aligned, 0.0)
    # entry_date 는 base.equity 의 맨 앞 원소다 (run_backtest 의 진입 시점) —
    # 오버레이가 수익만 바꿀 뿐 진입 시점 자체를 옮기지는 않으므로 그대로 물려받는다.
    equity = _equity_from_returns(returns, base.equity.index[0])
    return BacktestOutput(
        returns=returns,
        equity=equity,
        selections=base.selections.where(aligned, "CASH"),
        defensive_ratio=base.defensive_ratio,
    )


def run_with_tranches(
    spec: StrategySpec,
    daily: pd.DataFrame,
    n_tranches: int = 4,
    ma_overlay: bool = False,
    benchmark: str = "SPY",
    ma_days: int = 200,
    **kwargs: object,
) -> BacktestOutput:
    """자본을 `n_tranches` 로 나눠 서로 다른 주에 리밸런싱한 평균.

    단일 자산 + 월말 리밸런싱은 timing luck 에 취약하다 — 거래일 하루 차이로
    결과가 갈린다. **분산을 줄이는 장치이지 수익을 좇는 파라미터가 아니다.**

    `shift(-offset*5)` 를 원본 프레임에 바로 적용하면 뒤쪽 `offset*5`일이
    통째로 사라진다 — 오프셋이 클 때는 이 손실이 월 하나를 통째로 삼켜, 트랜치의
    유효 구간이 평이 기준보다 짧아진다(달의 경계를 넘는 절단이 조용히 발생).
    그래서 끝단에 마지막 관측치를 이월한 여분의 영업일을 붙여 시프트한 뒤,
    원래 날짜 범위로 다시 잘라 모든 트랜치가 같은 구간을 덮게 만든다.

    `ma_overlay=True` 면 이평 오버레이를 **트랜치마다 개별 적용한 뒤 평균**낸다
    (평균부터 낸 뒤 오버레이를 씌우는 순서가 아니다). 두 순서는 동치가 아니다 —
    평균 후 적용은 모든 슬리브를 한 덩어리로 취급해 같은 달에 다 같이 현금으로
    빠지거나 다 같이 투자 상태가 되고, 이는 "네 개의 독립적으로 리밸런싱되는
    슬리브"라는 트랜치의 전제 자체를 무너뜨린다. 슬리브마다 자신이 보는 가격
    경로(오프셋만큼 시프트된 것)로 자신의 이평 판정을 내리게 해야, 서로 다른
    주에 서로 다른 방어 타이밍을 갖는다는 트랜치의 취지가 오버레이와 결합해도
    유지된다.
    """
    pad = (n_tranches - 1) * 5
    if pad > 0:
        extra = pd.bdate_range(daily.index[-1], periods=pad + 1)[1:]
        padded = daily.reindex(daily.index.union(extra)).ffill()
    else:
        padded = daily

    outs = []
    for offset in range(n_tranches):
        shifted = padded.shift(-offset * 5).reindex(daily.index)
        if ma_overlay:
            outs.append(
                run_with_ma_overlay(spec, shifted, benchmark=benchmark, ma_days=ma_days, **kwargs)
            )
        else:
            outs.append(run_backtest(spec, shifted, **kwargs))  # type: ignore[arg-type]

    common = outs[0].returns.index
    for o in outs[1:]:
        common = common.intersection(o.returns.index)

    returns = sum(o.returns.reindex(common) for o in outs) / n_tranches
    assert isinstance(returns, pd.Series)
    returns.name = spec.name
    # entry_date 는 오프셋 0 트랜치(원본 daily 그대로, plain 과 같은 진입 시점)의
    # equity 맨 앞 원소를 물려받는다 — 트랜치 평균도 같은 시점에 같은 원금으로
    # 시작했다고 보는 것이 자연스럽다.
    equity = _equity_from_returns(returns, outs[0].equity.index[0])
    return BacktestOutput(
        returns=returns,
        equity=equity,
        selections=outs[0].selections.reindex(common),
        defensive_ratio=float(np.mean([o.defensive_ratio for o in outs])),
    )
