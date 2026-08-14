"""
횡단면 팩터 백테스트 엔진

체결 규약 (research/ic.py 의 forward_returns 와 동일해야 한다):
    신호: 리밸런싱일 t 의 종가까지의 정보
    체결: t+1 종가
    수익: t+1 → t+2 부터 새 비중으로 귀속

퀀트 관점:
- 세그먼트 사이 비중은 가격에 따라 드리프트한다. 이를 무시하고
  리밸런싱일 비중을 고정 적용하면 (특히 월간 이상 주기에서)
  수익률이 체계적으로 왜곡된다. 세그먼트별 누적곱으로 정확히 계산한다.
- 상장폐지 처리: 세그먼트 도중 가격이 사라진 종목은 마지막 관측가로
  청산했다고 가정하고 그 비중을 현금(수익률 0)으로 옮긴다.
  NaN 을 forward-fill 하면 폐지 종목이 '얼어붙은 가격'으로 살아남아
  생존 편향의 잔재가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from opt_portfolio.config import RISK_FREE_RATE
from opt_portfolio.factor.backtest.costs import CostModel
from opt_portfolio.factor.portfolio.weights import cap_sector_weights, compute_weights

#: 공분산 기반 비중 스킴이 요구하는 최소 트레일링 관측 일수
MIN_COV_OBS = 60


@dataclass(frozen=True)
class BacktestConfig:
    """전략 정의 — 이 객체의 필드들이 곧 PO 의 탐색 공간이다."""

    n_stocks: int = 20
    rebalance: str = "ME"  # 'ME' 월말 | 'QE' 분기말 | 'W-FRI' 주간
    weighting: str = "equal"
    max_weight: float = 0.10
    cov_window: int = 252
    cost: CostModel = field(default_factory=CostModel)
    ir_scale: float = 0.03  # MVO/BL 전용
    view_confidence: float = 0.5  # BL 전용
    #: 섹터 비중 상한. 1.0 이면 제약 없음. 팩터를 섹터 중립화해도 상위 N
    #: 선정 결과는 한 섹터에 몰릴 수 있고, 그건 의도하지 않은 매크로 베팅이다.
    max_sector_weight: float = 1.0
    #: 매매 유예구간. 상위 n_stocks 안에 들면 사고, n_stocks×hold_multiple
    #: 밖으로 밀려야 판다. 1.0 이면 밴드 없음(매 리밸런싱 상위 N 재선정).
    #: 순위가 문턱 근처에서 진동하는 종목을 반복 매매하는 낭비를 없앤다.
    hold_multiple: float = 1.0

    def __post_init__(self) -> None:
        if self.hold_multiple < 1.0:
            raise ValueError(
                f"hold_multiple 은 1.0 이상이어야 합니다 (받은 값 {self.hold_multiple}). "
                "청산 문턱이 진입보다 좁으면 밴드가 아니라 잡음이 된다."
            )


@dataclass
class BacktestResult:
    returns: pd.Series  # 일별 순수익률 (비용 차감 후)
    equity: pd.Series  # 누적 자산 (1.0 시작)
    holdings: pd.DataFrame  # 리밸런싱일 × 종목 목표비중
    turnover: pd.Series  # 리밸런싱일별 편도 회전율
    exposure: pd.Series  # 적용된 타이밍 익스포저

    def stats(
        self, periods_per_year: int = 252, risk_free_rate: float = RISK_FREE_RATE
    ) -> dict[str, float]:
        """
        성과 요약. Sharpe·Sortino 는 `walkforward.annualized_sharpe` 와 같은
        초과수익 규약(`config.RISK_FREE_RATE` 차감)을 따른다 — 규약이 갈리면
        `backtest` 와 `optimize` 의 숫자를 나란히 놓을 수 없다.
        """
        r = self.returns.dropna()
        if len(r) < 2:
            return {}
        ann = periods_per_year
        cum = float((1 + r).prod())
        years = len(r) / ann
        vol = float(r.std(ddof=1) * np.sqrt(ann))
        excess = r - risk_free_rate / ann
        downside = r[r < 0].std(ddof=1) * np.sqrt(ann) if (r < 0).any() else np.nan
        dd = (self.equity / self.equity.cummax() - 1.0).min()
        cagr = cum ** (1 / years) - 1 if years > 0 else np.nan
        return {
            "total_return": cum - 1,
            "cagr": cagr,
            "ann_vol": vol,
            "sharpe": float(excess.mean() / r.std(ddof=1) * np.sqrt(ann)) if vol > 0 else np.nan,
            "sortino": (
                float(excess.mean() * ann / downside) if downside and downside > 0 else np.nan
            ),
            "max_drawdown": float(dd),
            "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "avg_turnover": float(self.turnover.mean()),
            "n_rebalances": int(len(self.turnover)),
        }


def rebalance_dates(calendar: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """캘린더에서 주기별 마지막 거래일을 뽑는다."""
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.resample(freq).last().dropna().to_numpy())


def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    config: BacktestConfig,
    *,
    universe: pd.DataFrame | None = None,
    market_caps: pd.DataFrame | None = None,
    exposure: pd.Series | None = None,
    sectors: pd.Series | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> BacktestResult:
    """
    Args:
        close: 일별 수정 종가 (date × ticker). 상장폐지 종목은 폐지 이후 NaN.
        scores: 합성 스코어 패널 — composite_score() 결과. 클수록 좋음.
        universe: 불리언 마스크 (date × ticker). 없으면 스코어 관측 = 편입 가능.
        exposure: 타이밍 익스포저 (판정일 기준). 내부에서 1일 시프트해 적용.
        start/end: 평가 구간. 신호·공분산은 구간 밖 과거를 쓸 수 있다.
    """
    calendar = close.index
    # fill_method=None: 상장폐지 후 NaN 이 pad 로 되살아나는 것을 막는다
    rets = close.pct_change(fill_method=None)

    span = calendar
    if start is not None:
        span = span[span >= start]
    if end is not None:
        span = span[span <= end]
    if len(span) < 3:
        raise ValueError("평가 구간이 너무 짧습니다")

    rb_dates = rebalance_dates(pd.DatetimeIndex(span), config.rebalance)
    rb_dates = rb_dates[rb_dates < span[-1]]  # 마지막 날 신호는 체결 불가

    # 익스포저: 판정일 t → 적용 t+1 (신호와 같은 규약)
    exp_applied = (
        exposure.reindex(calendar).ffill().shift(1).fillna(1.0)
        if exposure is not None
        else pd.Series(1.0, index=calendar)
    )

    port_ret = pd.Series(0.0, index=span)
    holdings_log: dict[pd.Timestamp, pd.Series] = {}
    turnover_log: dict[pd.Timestamp, float] = {}
    current_w = pd.Series(dtype=float)  # 직전 세그먼트 종료 시점의 드리프트 비중

    held: pd.Index = pd.Index([])  # 직전 리밸런싱에서 담은 종목 (밴드 판정용)

    for i, signal_date in enumerate(rb_dates):
        # 밴드를 적용하려면 후보를 넉넉히 뽑아야 한다 — n 만 뽑으면 밴드 안에
        # 남아 있는 보유 종목이 후보에서 이미 잘려 나간다.
        pool_size = max(config.n_stocks, int(config.n_stocks * config.hold_multiple))
        pool = _select(scores, close, universe, signal_date, pool_size)
        if pool.empty:
            continue
        selected = select_with_band(pool, held, config.n_stocks, config.hold_multiple)
        if selected.empty:
            continue
        held = selected.index

        new_w = _weights_for(selected, rets, market_caps, signal_date, config, sectors)

        # 체결일: 신호 다음 거래일
        pos = calendar.searchsorted(signal_date) + 1
        if pos >= len(calendar):
            break
        exec_date = calendar[pos]

        # 세그먼트: 체결 다음날부터 다음 체결일까지 새 비중으로 수익 귀속
        next_signal = rb_dates[i + 1] if i + 1 < len(rb_dates) else span[-1]
        next_pos = min(calendar.searchsorted(next_signal) + 1, len(calendar) - 1)
        seg = calendar[(calendar > exec_date) & (calendar <= calendar[next_pos])]

        traded = new_w.sub(current_w, fill_value=0.0).abs()
        turnover_log[signal_date] = float(traded.sum()) / 2.0
        holdings_log[signal_date] = new_w

        # 비용은 체결일 수익률에서 차감
        if exec_date in port_ret.index:
            port_ret.loc[exec_date] -= config.cost.rebalance_cost(traded)

        if len(seg) == 0:
            current_w = new_w
            continue

        seg_ret, current_w = _drift_segment(new_w, rets.loc[seg])
        common = seg_ret.index.intersection(port_ret.index)
        port_ret.loc[common] += seg_ret.loc[common]

    net = port_ret * exp_applied.reindex(span).to_numpy()
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        returns=net,
        equity=equity,
        holdings=pd.DataFrame(holdings_log).T.fillna(0.0),
        turnover=pd.Series(turnover_log, dtype=float),
        exposure=exp_applied.reindex(span),
    )


# ------------------------------------------------------------------ 내부


def select_with_band(
    row: pd.Series,
    held: list[str] | pd.Index,
    n: int,
    hold_multiple: float = 1.0,
) -> pd.Series:
    """
    매매 유예구간을 적용한 상위 n 선정.

    보유 종목은 n×hold_multiple 위 안에 있으면 유지하고, 남는 자리를
    상위 신규 종목으로 채운다. `hold_multiple=1.0` 이면 단순 상위 n 이다.
    """
    if hold_multiple <= 1.0 or len(held) == 0:
        return row.nlargest(n)

    keep_rank = int(round(n * hold_multiple))
    incumbents = row.nlargest(keep_rank).index.intersection(pd.Index(held))
    incumbents = incumbents[:n]  # 정원 초과 시 스코어 높은 쪽 우선

    slots = n - len(incumbents)
    if slots <= 0:
        return row.loc[incumbents]
    challengers = row.drop(index=incumbents, errors="ignore").nlargest(slots)
    return row.loc[incumbents.union(challengers.index)].sort_values(ascending=False)


def _select(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    universe: pd.DataFrame | None,
    date: pd.Timestamp,
    n: int,
) -> pd.Series:
    """신호일 기준 상위 n 종목의 스코어. 거래 가능(가격 존재) 종목만."""
    if date not in scores.index:
        loc = scores.index.searchsorted(date, side="right") - 1
        if loc < 0:
            return pd.Series(dtype=float)
        date_eff = scores.index[loc]
    else:
        date_eff = date

    row = scores.loc[date_eff].dropna()
    tradeable = close.loc[date].dropna().index
    row = row[row.index.isin(tradeable)]

    if universe is not None and date in universe.index:
        allowed = universe.loc[date]
        row = row[row.index.isin(allowed.index[allowed.fillna(False)])]

    return row.nlargest(n)


def _weights_for(
    selected: pd.Series,
    rets: pd.DataFrame,
    market_caps: pd.DataFrame | None,
    date: pd.Timestamp,
    config: BacktestConfig,
    sectors: pd.Series | None = None,
) -> pd.Series:
    names = selected.index
    window = rets.loc[:date, names].tail(config.cov_window)

    # 공분산 스킴에 관측이 부족하면 동일가중 폴백 (에러로 죽지 않는다 —
    # walk-forward 초기 구간에서 전체 실험이 무산되는 것을 막기 위함)
    scheme = config.weighting
    if scheme != "equal" and window.dropna(how="any").shape[0] < MIN_COV_OBS:
        scheme = "equal"

    caps = None
    if market_caps is not None and date in market_caps.index:
        caps = market_caps.loc[date, market_caps.columns.intersection(names)]

    kwargs = {}
    if scheme in ("mvo", "black_litterman"):
        kwargs["ir_scale"] = config.ir_scale
    if scheme == "black_litterman":
        kwargs["view_confidence"] = config.view_confidence

    weights = (
        compute_weights(
            scheme,
            window.dropna(how="any"),
            selected,
            max_weight=config.max_weight,
            market_caps=caps,
            **kwargs,
        )
        .reindex(names)
        .fillna(1.0 / len(names) if scheme == "equal" else 0.0)
    )
    # 섹터 상한은 비중 스킴과 무관하게 마지막에 적용한다 — 어떤 스킴을 쓰든
    # 쏠림은 같은 방식으로 막아야 한다.
    if config.max_sector_weight < 1.0 and sectors is not None:
        weights = cap_sector_weights(weights, sectors, config.max_sector_weight)
    return weights


def _drift_segment(
    w0: pd.Series,
    seg_rets: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    세그먼트 내 일별 포트폴리오 수익률과 종료 시점 드리프트 비중.

    상장폐지(NaN 수익률) 종목은 그 시점부터 수익률 0 (현금 청산 가정).
    """
    r = seg_rets[w0.index].fillna(0.0)
    growth = (1.0 + r).cumprod()
    value = growth.mul(w0, axis=1).sum(axis=1)  # 포트폴리오 가치 (시작 1.0)

    port_ret = value.pct_change()
    port_ret.iloc[0] = value.iloc[0] - 1.0

    end_w = growth.iloc[-1] * w0 / value.iloc[-1]
    return port_ret, end_w
