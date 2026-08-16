# 전술적 자산배분(TAA) 재설계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** VAA 를 대체할 전술적 자산배분을 찾되, 사전 등록한 9개 구성을 PBO/DSR 관문에 태워 **채택 없음이라는 결론도 낼 수 있게** 만든다.

**Architecture:** 새 패키지 `src/opt_portfolio/taa/` 에 선언적 전략 스펙 + 월별 리밸런싱 엔진을 만든다. 전략은 코드가 아니라 `StrategySpec` 데이터로 표현되므로 VAA·BAA·변형이 같은 엔진을 탄다. 검증은 팩터 엔진의 `factor.research.overfitting` 을 재사용한다.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, ruff, mypy. 데이터는 로컬 Sharadar 펀드 벌크 zip.

**Spec:** `docs/superpowers/specs/2026-08-17-taa-strategy-design.md`

## Global Constraints

- **가격은 `closeadj`(배당조정) 를 쓴다.** 실측 확인: TLT 2010-01-04 이 `close`=89.81 / `closeadj`=55.163 이다. `close` 를 쓰면 16년간 −8.6%, `closeadj` 면 +48.7% 로 **채권 ETF 수익이 뒤집힌다.**
- **DSR 에는 월별 수익률을 그대로 넘긴다 — 연율화 금지.** `deflated_sharpe_ratio` docstring 의 요구사항이다. 이 저장소는 연율화 주기를 이미 두 번 틀렸다.
- **사전 등록 9개 구성을 늘리지 않는다.** `n_trials = 9` 는 고정이다.
- line-length 100, ruff, mypy `disallow_untyped_defs` — **신규 함수에 타입 힌트 필수**.
- 커밋 메시지는 한국어 Conventional Commits. 본문에 **왜 그게 문제였는지**를 쓴다.
- 벌크 zip 경로 기본값: `~/data/sharadar/raw/funds.csv.zip`.
- 신규 코드는 `src/opt_portfolio/taa/` 이하에만 쓴다. 허용 의존은 `factor.research.overfitting` 과 `analysis.metrics` 둘뿐이다.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/opt_portfolio/taa/__init__.py` | 공개 API 재수출 |
| `src/opt_portfolio/taa/data.py` | 벌크 zip → 일별 가격 패널 (`closeadj`) |
| `src/opt_portfolio/taa/signals.py` | 13612W · SMA 비율 — 순수 함수 |
| `src/opt_portfolio/taa/strategy.py` | `StrategySpec` 선언 + 월별 선택 로직 |
| `src/opt_portfolio/taa/backtest.py` | 월별 리밸런싱 엔진 · 비용 |
| `src/opt_portfolio/taa/registry.py` | 사전 등록 9개 구성 |
| `scripts/run_taa.py` | 9개 실행 → PBO/DSR → 표 출력 |
| `tests/taa/test_*.py` | 각 모듈 대응 |

---

## Task 1: 데이터 어댑터

**Files:**
- Create: `src/opt_portfolio/taa/__init__.py`, `src/opt_portfolio/taa/data.py`
- Test: `tests/taa/test_data.py`

**Interfaces:**
- Consumes: 없음
- Produces: `load_prices(tickers: list[str], zip_path: Path | None = None) -> pd.DataFrame` — 일별 종가 패널, 인덱스 `DatetimeIndex`, 컬럼 = 티커, 값 = `closeadj`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# tests/taa/test_data.py
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from opt_portfolio.taa.data import load_prices

HEADER = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated"


def _make_zip(tmp_path: Path, rows: list[str]) -> Path:
    csv = tmp_path / "funds.csv"
    csv.write_text("\n".join([HEADER, *rows]) + "\n")
    zp = tmp_path / "funds.csv.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(csv, arcname="funds.csv")
    return zp


class TestLoadPrices:
    def test_uses_closeadj_not_close(self, tmp_path: Path) -> None:
        """배당조정가를 써야 한다 — close 를 쓰면 채권 ETF 수익이 뒤집힌다."""
        zp = _make_zip(tmp_path, [
            "TLT,2010-01-04,0,0,0,89.81,100,55.163,89.81,2026-08-14",
            "TLT,2026-08-14,0,0,0,82.04,100,82.04,82.04,2026-08-14",
        ])
        px = load_prices(["TLT"], zip_path=zp)

        assert px.loc[pd.Timestamp("2010-01-04"), "TLT"] == pytest.approx(55.163)
        assert px.loc[pd.Timestamp("2026-08-14"), "TLT"] == pytest.approx(82.04)

    def test_filters_to_requested_tickers(self, tmp_path: Path) -> None:
        zp = _make_zip(tmp_path, [
            "TLT,2010-01-04,0,0,0,1,100,55.163,1,2026-08-14",
            "SPY,2010-01-04,0,0,0,1,100,90.0,1,2026-08-14",
            "QQQ,2010-01-04,0,0,0,1,100,40.0,1,2026-08-14",
        ])
        px = load_prices(["TLT", "SPY"], zip_path=zp)

        assert sorted(px.columns) == ["SPY", "TLT"]

    def test_missing_ticker_fails_loudly(self, tmp_path: Path) -> None:
        """조용히 빈 컬럼을 만들지 않는다 — 이 저장소의 지배적 실패 유형이다."""
        zp = _make_zip(tmp_path, ["SPY,2010-01-04,0,0,0,1,100,90.0,1,2026-08-14"])

        with pytest.raises(ValueError, match="NOPE"):
            load_prices(["SPY", "NOPE"], zip_path=zp)

    def test_index_is_sorted_datetime(self, tmp_path: Path) -> None:
        zp = _make_zip(tmp_path, [
            "SPY,2010-01-05,0,0,0,1,100,91.0,1,2026-08-14",
            "SPY,2010-01-04,0,0,0,1,100,90.0,1,2026-08-14",
        ])
        px = load_prices(["SPY"], zip_path=zp)

        assert isinstance(px.index, pd.DatetimeIndex)
        assert px.index.is_monotonic_increasing
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/taa/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'opt_portfolio.taa'`

- [ ] **Step 3: 최소 구현**

```python
# src/opt_portfolio/taa/__init__.py
"""전술적 자산배분(TAA) — ETF 월별 로테이션.

`factor/`(개별주 팩터) 와도 `strategies/`(기존 VAA) 와도 분리된 새 서브시스템이다.
설계 근거는 `docs/superpowers/specs/2026-08-17-taa-strategy-design.md`.
"""

from .data import load_prices

__all__ = ["load_prices"]
```

```python
# src/opt_portfolio/taa/data.py
"""Sharadar 펀드 벌크 → 가격 패널.

**`closeadj`(배당조정) 를 쓴다.** 실측: TLT 2010-01-04 이 close 89.81 /
closeadj 55.163 이다. close 를 쓰면 16년간 −8.6%, closeadj 면 +48.7% 로
채권 ETF 수익이 통째로 뒤집힌다. 이 전략은 시간의 절반을 채권에 머무르므로
치명적이다.

yfinance 가 아니라 로컬 벌크를 쓰는 이유: 구독을 종료해도 남고, 팩터 엔진과
원본이 같아 두 서브시스템의 성과가 비교 가능해진다.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_ZIP = Path.home() / "data/sharadar/raw/funds.csv.zip"

#: 벌크에서 읽는 가격 컬럼. 바꾸지 말 것 — 모듈 docstring 참조.
PRICE_COLUMN = "closeadj"


def load_prices(tickers: list[str], zip_path: Path | None = None) -> pd.DataFrame:
    """요청한 티커의 일별 배당조정 종가 패널.

    Args:
        tickers: 티커 목록
        zip_path: 펀드 벌크 zip (기본 `~/data/sharadar/raw/funds.csv.zip`)

    Returns:
        인덱스 = 거래일(오름차순), 컬럼 = 티커, 값 = `closeadj`

    Raises:
        ValueError: 요청한 티커 중 하나라도 벌크에 없으면. 조용히 빈 컬럼을
            만들면 이후 모든 계산이 NaN 으로 흘러가 결과가 "성공" 으로 보인다.
    """
    path = zip_path or DEFAULT_ZIP
    if not path.exists():
        raise FileNotFoundError(f"펀드 벌크가 없다: {path}")

    wanted = set(tickers)
    rows: list[tuple[str, str, float]] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"zip 안의 csv 가 하나가 아니다: {names}")
        with zf.open(names[0]) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
            for row in reader:
                t = row["ticker"]
                if t not in wanted:
                    continue
                raw = row[PRICE_COLUMN]
                if raw == "":
                    continue
                rows.append((t, row["date"], float(raw)))

    if not rows:
        raise ValueError(f"벌크에서 아무 행도 못 찾았다: {sorted(wanted)}")

    frame = pd.DataFrame(rows, columns=["ticker", "date", "px"])
    panel = frame.pivot(index="date", columns="ticker", values="px")
    panel.index = pd.DatetimeIndex(panel.index)
    panel = panel.sort_index()

    missing = wanted - set(panel.columns)
    if missing:
        raise ValueError(f"벌크에 없는 티커: {sorted(missing)}")

    return panel[sorted(wanted)]
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/taa/test_data.py -v`
Expected: PASS (4개)

- [ ] **Step 5: 실데이터로 한 번 확인한다**

Run:
```bash
uv run python -c "
from opt_portfolio.taa.data import load_prices
px = load_prices(['SPY','TLT','BIL'])
print(px.shape, px.index.min().date(), px.index.max().date())
print(px.tail(1))
"
```
Expected: BIL 상장(2007-05-30) 이전 행은 `NaN`, 최신일 2026-08-14. **행 수가 0 이거나 최신일이 2026-08-14 가 아니면 멈추고 원인을 찾는다.**

- [ ] **Step 6: 커밋**

```bash
git add src/opt_portfolio/taa/ tests/taa/test_data.py
git commit -m "feat(taa): 펀드 벌크 어댑터 — closeadj 를 쓴다

close 를 쓰면 TLT 가 16년간 -8.6%, closeadj 면 +48.7% 다. 이 전략은 시간의
절반을 채권에 머무르므로 잘못 고르면 결과가 통째로 뒤집힌다. 실측으로 확인하고
상수로 못박았다.

없는 티커는 예외로 죽인다 — 조용히 빈 컬럼을 만들면 NaN 이 흘러가 '성공' 으로
보인다."
```

---

## Task 2: 모멘텀 시그널

**Files:**
- Create: `src/opt_portfolio/taa/signals.py`
- Test: `tests/taa/test_signals.py`

**Interfaces:**
- Consumes: Task 1 의 가격 패널
- Produces:
  - `to_monthly(daily: pd.DataFrame) -> pd.DataFrame` — 월말 종가
  - `momentum_13612w(monthly: pd.DataFrame) -> pd.DataFrame`
  - `sma_ratio(monthly: pd.DataFrame, window: int = 13) -> pd.DataFrame`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# tests/taa/test_signals.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.taa.signals import momentum_13612w, sma_ratio, to_monthly


def _monthly(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    return pd.DataFrame({"A": start + step * np.arange(n)}, index=idx)


class TestToMonthly:
    def test_takes_last_observation_of_month(self) -> None:
        idx = pd.date_range("2010-01-01", periods=45, freq="D")
        daily = pd.DataFrame({"A": np.arange(45, dtype=float)}, index=idx)

        m = to_monthly(daily)

        assert m.loc[pd.Timestamp("2010-01-31"), "A"] == 30.0


class TestMomentum13612W:
    def test_matches_hand_computation(self) -> None:
        """12*r1 + 4*r3 + 2*r6 + 1*r12 — 논문 정의 그대로."""
        m = _monthly(13)
        mom = momentum_13612w(m)

        p = m["A"]
        expected = (
            12 * (p.iloc[12] / p.iloc[11] - 1)
            + 4 * (p.iloc[12] / p.iloc[9] - 1)
            + 2 * (p.iloc[12] / p.iloc[6] - 1)
            + 1 * (p.iloc[12] / p.iloc[0] - 1)
        )
        assert mom["A"].iloc[-1] == pytest.approx(expected)

    def test_needs_twelve_months_of_history(self) -> None:
        """12개월이 안 차면 NaN — 없는 데이터로 판정하지 않는다."""
        mom = momentum_13612w(_monthly(12))

        assert mom["A"].isna().all()

    def test_negative_when_price_falls(self) -> None:
        m = _monthly(13, start=200.0, step=-5.0)

        assert momentum_13612w(m)["A"].iloc[-1] < 0


class TestSmaRatio:
    def test_is_price_over_trailing_average(self) -> None:
        m = _monthly(13)
        ratio = sma_ratio(m, window=13)

        p = m["A"]
        assert ratio["A"].iloc[-1] == pytest.approx(p.iloc[-1] / p.iloc[:13].mean())

    def test_above_one_in_uptrend_below_in_downtrend(self) -> None:
        assert sma_ratio(_monthly(13), 13)["A"].iloc[-1] > 1.0
        assert sma_ratio(_monthly(13, 200.0, -5.0), 13)["A"].iloc[-1] < 1.0
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/taa/test_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: opt_portfolio.taa.signals`

- [ ] **Step 3: 최소 구현**

```python
# src/opt_portfolio/taa/signals.py
"""모멘텀 시그널 — 순수 함수.

두 지표를 쓰는 이유가 다르다. **경보는 빠르게, 선택은 느리게** 라는 것이
BAA 논문의 표현이다.

- `momentum_13612w` : 카나리아(위험 경보) 판정용. 최근 1개월에 무게가 실린다
- `sma_ratio`       : 자산 선택용. 13개월 평균 대비라 훨씬 느리다

13612W 의 가중치 12/4/2/1 은 임의가 아니라 **연율화 계수**다 — 1개월 수익 ×12,
3개월 ×4, 6개월 ×2, 12개월 ×1 로 서로 다른 시간축의 연율 수익을 합한 값이다.
"""

from __future__ import annotations

import pandas as pd

#: (개월 수, 가중치) — Keller 13612W
_MOMENTUM_TERMS: tuple[tuple[int, int], ...] = ((1, 12), (3, 4), (6, 2), (12, 1))


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """일별 패널 → 월말 종가."""
    return daily.resample("ME").last()


def momentum_13612w(monthly: pd.DataFrame) -> pd.DataFrame:
    """13612W 모멘텀. 12개월 미만 구간은 NaN 이다."""
    score = None
    for months, weight in _MOMENTUM_TERMS:
        term = weight * monthly.pct_change(months)
        score = term if score is None else score + term
    assert score is not None  # _MOMENTUM_TERMS 가 비지 않음
    return score


def sma_ratio(monthly: pd.DataFrame, window: int = 13) -> pd.DataFrame:
    """현재가 / 직전 `window` 개월 평균. 1 보다 크면 상승 추세."""
    return monthly / monthly.rolling(window, min_periods=window).mean()
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/taa/test_signals.py -v`
Expected: PASS (6개)

- [ ] **Step 5: 커밋**

```bash
git add src/opt_portfolio/taa/signals.py tests/taa/test_signals.py
git commit -m "feat(taa): 13612W · SMA 비율 시그널

두 지표를 쓰는 이유가 다르다 — 경보는 빠르게(13612W), 선택은 느리게(SMA13).
BAA 가 VAA 와 갈리는 지점이 여기다. VAA 는 둘 다 13612W 로 해서 경보가
과민했다.

12개월이 안 차면 NaN 을 낸다. 없는 데이터로 판정하면 초기 구간이 조용히
틀린다."
```

---

## Task 3: 전략 스펙과 선택 로직

**Files:**
- Create: `src/opt_portfolio/taa/strategy.py`
- Test: `tests/taa/test_strategy.py`

**Interfaces:**
- Consumes: Task 2 의 `momentum_13612w`, `sma_ratio`
- Produces:
  - `StrategySpec` (frozen dataclass): `name: str`, `canary: tuple[str, ...]`, `offensive: tuple[str, ...]`, `defensive: tuple[str, ...]`, `top_n_offensive: int`, `top_n_defensive: int`, `selection: str` (`"13612w"` | `"sma13"`), `cash_ticker: str | None`, `static_weights: dict[str, float] | None`
  - `select_weights(spec: StrategySpec, mom: pd.DataFrame, sel: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# tests/taa/test_strategy.py
from __future__ import annotations

import pandas as pd
import pytest

from opt_portfolio.taa.strategy import StrategySpec, select_weights

D = pd.Timestamp("2020-06-30")


def _frames(mom: dict[str, float], sel: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(mom, index=[D]),
        pd.DataFrame(sel, index=[D]),
    )


SPEC = StrategySpec(
    name="test",
    canary=("SPY", "EEM"),
    offensive=("QQQ", "EEM"),
    defensive=("IEF", "BIL"),
    top_n_offensive=1,
    top_n_defensive=1,
    selection="sma13",
    cash_ticker="BIL",
)


class TestSelectWeights:
    def test_all_canary_positive_goes_offensive(self) -> None:
        mom, sel = _frames(
            {"SPY": 0.5, "EEM": 0.1, "QQQ": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.20, "EEM": 1.05, "IEF": 1.01, "BIL": 1.00},
        )
        w = select_weights(SPEC, mom, sel, D)

        assert w == {"QQQ": 1.0}

    def test_any_canary_negative_goes_defensive(self) -> None:
        """VAA 도 BAA 도 공통인 breadth 규칙."""
        mom, sel = _frames(
            {"SPY": 0.5, "EEM": -0.01, "QQQ": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.20, "EEM": 1.05, "IEF": 1.10, "BIL": 1.00},
        )
        w = select_weights(SPEC, mom, sel, D)

        assert w == {"IEF": 1.0}

    def test_defensive_asset_below_cash_is_replaced_by_cash(self) -> None:
        """BIL 을 못 이기면 현금 — SHY 를 연 0.05% 로 들고 있던 문제의 해법."""
        mom, sel = _frames(
            {"SPY": -0.1, "EEM": 0.1, "QQQ": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.20, "EEM": 1.05, "IEF": 0.95, "BIL": 1.00},
        )
        w = select_weights(SPEC, mom, sel, D)

        assert w == {"BIL": 1.0}

    def test_multiple_holdings_are_equal_weighted(self) -> None:
        spec = StrategySpec(
            name="t2", canary=("SPY",), offensive=("QQQ", "EEM", "IEF"),
            defensive=("BIL",), top_n_offensive=2, top_n_defensive=1,
            selection="sma13", cash_ticker="BIL",
        )
        mom, sel = _frames(
            {"SPY": 0.5, "QQQ": 0.0, "EEM": 0.0, "IEF": 0.0, "BIL": 0.0},
            {"QQQ": 1.30, "EEM": 1.20, "IEF": 1.10, "BIL": 1.00},
        )
        w = select_weights(spec, mom, sel, D)

        assert w == {"QQQ": 0.5, "EEM": 0.5}

    def test_static_spec_ignores_signals(self) -> None:
        """60/40 기준선 — 아무 판단도 하지 않는다."""
        spec = StrategySpec(
            name="60/40", canary=(), offensive=(), defensive=(),
            top_n_offensive=0, top_n_defensive=0, selection="sma13",
            cash_ticker=None, static_weights={"SPY": 0.6, "IEF": 0.4},
        )
        mom, sel = _frames({"SPY": -9.0}, {"SPY": 0.1})

        assert select_weights(spec, mom, sel, D) == {"SPY": 0.6, "IEF": 0.4}

    def test_uses_13612w_when_selection_is_13612w(self) -> None:
        """VAA 는 선택도 13612W 로 한다 — BAA 와 갈리는 지점."""
        spec = StrategySpec(
            name="vaa", canary=("SPY",), offensive=("QQQ", "EEM"),
            defensive=("IEF",), top_n_offensive=1, top_n_defensive=1,
            selection="13612w", cash_ticker=None,
        )
        mom, sel = _frames(
            {"SPY": 0.5, "QQQ": 0.1, "EEM": 0.9, "IEF": 0.0},
            {"QQQ": 9.9, "EEM": 0.1, "IEF": 1.0},   # sma 는 QQQ 가 높지만
        )
        w = select_weights(spec, mom, sel, D)

        assert w == {"EEM": 1.0}   # 13612W 기준이라 EEM
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/taa/test_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError: opt_portfolio.taa.strategy`

- [ ] **Step 3: 최소 구현**

```python
# src/opt_portfolio/taa/strategy.py
"""전략을 데이터로 선언한다.

VAA·BAA·정적 배분이 **같은 엔진을 타야** 비교가 성립한다. 그래서 전략을
코드가 아니라 `StrategySpec` 으로 표현한다 — 팩터 엔진이 전략을 JSON 하나로
선언하는 것과 같은 이유다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """전략 하나의 완전한 선언.

    Attributes:
        canary: 위험 경보용 자산. **투자 대상과 분리한다** — VAA 는 이 둘을
            겸해서, 살 생각도 없는 EEM·EFA 약세가 포트폴리오를 방어로 밀어냈다.
        offensive: 위험 국면이 아닐 때 고를 후보
        defensive: 위험 국면에 고를 후보
        selection: 선택 지표. `"13612w"`(VAA) 또는 `"sma13"`(BAA)
        cash_ticker: 지정하면 방어 자산이 이것보다 못할 때 현금으로 대체한다
            (dual momentum). `None` 이면 비활성.
        static_weights: 지정하면 시그널을 무시하고 이 비중을 유지한다 (기준선용)
    """

    name: str
    canary: tuple[str, ...]
    offensive: tuple[str, ...]
    defensive: tuple[str, ...]
    top_n_offensive: int
    top_n_defensive: int
    selection: str = "sma13"
    cash_ticker: str | None = None
    static_weights: dict[str, float] | None = None

    def tickers(self) -> list[str]:
        """이 전략이 필요로 하는 전체 티커."""
        names = set(self.canary) | set(self.offensive) | set(self.defensive)
        if self.cash_ticker:
            names.add(self.cash_ticker)
        if self.static_weights:
            names |= set(self.static_weights)
        return sorted(names)


def is_defensive(spec: StrategySpec, mom: pd.DataFrame, date: pd.Timestamp) -> bool:
    """카나리아 중 **하나라도** 모멘텀이 음수면 방어 (breadth 규칙)."""
    if not spec.canary:
        return False
    scores = mom.loc[date, list(spec.canary)]
    return bool((scores < 0).any())


def select_weights(
    spec: StrategySpec,
    mom: pd.DataFrame,
    sel: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """해당 시점의 목표 비중. 합은 1.0 (전액 현금 대체 시에도)."""
    if spec.static_weights is not None:
        return dict(spec.static_weights)

    metric = mom if spec.selection == "13612w" else sel
    defensive = is_defensive(spec, mom, date)
    pool = list(spec.defensive) if defensive else list(spec.offensive)
    top_n = spec.top_n_defensive if defensive else spec.top_n_offensive

    ranked = metric.loc[date, pool].sort_values(ascending=False)
    picks = list(ranked.index[:top_n])

    # dual momentum — 현금을 못 이기는 방어 자산은 현금으로 바꾼다.
    # SHY 를 연 0.05% 로 24% 기간 들고 있던 문제의 해법이다.
    if defensive and spec.cash_ticker:
        cash_score = metric.loc[date, spec.cash_ticker]
        picks = [t if metric.loc[date, t] > cash_score else spec.cash_ticker for t in picks]

    weight = 1.0 / len(picks)
    out: dict[str, float] = {}
    for t in picks:
        out[t] = out.get(t, 0.0) + weight
    return out
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/taa/test_strategy.py -v`
Expected: PASS (6개)

- [ ] **Step 5: 커밋**

```bash
git add src/opt_portfolio/taa/strategy.py tests/taa/test_strategy.py
git commit -m "feat(taa): 전략을 데이터로 선언한다

VAA·BAA·정적 배분이 같은 엔진을 타야 비교가 성립한다. 그래서 전략을 코드가
아니라 StrategySpec 으로 표현한다.

핵심은 canary 를 offensive 와 분리한 것이다. VAA 는 공격 4자산이 투자
대상이자 경보기를 겸해서, 살 생각도 없는 EEM·EFA 약세가 포트폴리오 전체를
방어로 밀어냈다 — 2011~2026 에 방어 55%, 그중 SHY 가 연 0.05% 였다.

cash_ticker 대비 dual momentum 이 그 SHY 문제를 직접 겨냥한다."
```

---

## Task 4: 월별 리밸런싱 엔진

**Files:**
- Create: `src/opt_portfolio/taa/backtest.py`
- Test: `tests/taa/test_backtest.py`

**Interfaces:**
- Consumes: Task 1~3 전부
- Produces:
  - `BacktestOutput` (frozen dataclass): `returns: pd.Series` (월별), `equity: pd.Series`, `selections: pd.Series` (쉼표 구분 티커), `defensive_ratio: float`
  - `run_backtest(spec, daily, start=None, end=None, cost_bps=10.0) -> BacktestOutput`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# tests/taa/test_backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opt_portfolio.taa.backtest import run_backtest
from opt_portfolio.taa.strategy import StrategySpec

STATIC = StrategySpec(
    name="allspy", canary=(), offensive=(), defensive=(),
    top_n_offensive=0, top_n_defensive=0, static_weights={"SPY": 1.0},
)


def _daily(n_months: int = 40, monthly_step: float = 0.01) -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", periods=n_months * 21, freq="B")
    ramp = (1 + monthly_step) ** (np.arange(len(idx)) / 21)
    return pd.DataFrame({"SPY": 100.0 * ramp, "IEF": 100.0 * np.ones(len(idx))}, index=idx)


class TestRunBacktest:
    def test_static_full_allocation_tracks_the_asset(self) -> None:
        daily = _daily()
        out = run_backtest(STATIC, daily, cost_bps=0.0)

        monthly = daily["SPY"].resample("ME").last()
        expected = monthly.iloc[-1] / monthly.iloc[len(monthly) - len(out.returns) - 1] - 1
        assert (1 + out.returns).prod() - 1 == pytest.approx(expected, rel=1e-6)

    def test_costs_reduce_returns(self) -> None:
        daily = _daily()
        free = run_backtest(STATIC, daily, cost_bps=0.0)
        costly = run_backtest(STATIC, daily, cost_bps=100.0)

        assert costly.equity.iloc[-1] < free.equity.iloc[-1]

    def test_static_allocation_pays_no_turnover_cost_after_entry(self) -> None:
        """비중이 안 바뀌면 회전이 없다 — 비용을 매달 물리면 안 된다."""
        daily = _daily()
        free = run_backtest(STATIC, daily, cost_bps=0.0)
        costly = run_backtest(STATIC, daily, cost_bps=100.0)

        gap = free.equity.iloc[-1] - costly.equity.iloc[-1]
        assert gap < free.equity.iloc[-1] * 0.02

    def test_no_lookahead_signal_uses_prior_month_close(self) -> None:
        """t월말 신호로 t+1월 수익을 얻는다. 같은 달 수익을 쓰면 룩어헤드다."""
        daily = _daily()
        out = run_backtest(STATIC, daily, cost_bps=0.0)
        monthly = daily["SPY"].resample("ME").last()

        assert out.returns.index[-1] == monthly.index[-1]
        assert len(out.returns) < len(monthly)

    def test_equity_and_returns_are_consistent(self) -> None:
        out = run_backtest(STATIC, _daily(), cost_bps=10.0)

        assert out.equity.iloc[-1] == pytest.approx(
            out.equity.iloc[0] * (1 + out.returns).prod(), rel=1e-9
        )

    def test_defensive_ratio_is_zero_for_static(self) -> None:
        assert run_backtest(STATIC, _daily(), cost_bps=0.0).defensive_ratio == 0.0
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/taa/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: opt_portfolio.taa.backtest`

- [ ] **Step 3: 최소 구현**

```python
# src/opt_portfolio/taa/backtest.py
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
    returns: pd.Series
    equity: pd.Series
    selections: pd.Series
    defensive_ratio: float


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
    fwd = monthly.pct_change().shift(-1)

    needed = spec.tickers()
    usable = mom.dropna(how="any", subset=needed).index
    if start is not None:
        usable = usable[usable >= start]
    if end is not None:
        usable = usable[usable <= end]
    if len(usable) == 0:
        raise ValueError(f"[{spec.name}] 평가 가능한 시점이 없다 — 데이터 구간을 확인하라")

    prev: dict[str, float] = {}
    dates, rets, picks, defensive_flags = [], [], [], []

    for date in usable:
        nxt = fwd.loc[date]
        weights = select_weights(spec, mom, sel, date)
        gross = float(sum(w * nxt[t] for t, w in weights.items()))
        if not np.isfinite(gross):
            continue  # 다음 달 가격이 없는 마지막 시점

        turnover = sum(abs(weights.get(t, 0.0) - prev.get(t, 0.0)) for t in set(weights) | set(prev))
        cost = turnover * cost_bps / 10_000.0

        dates.append(date)
        rets.append(gross - cost)
        picks.append(",".join(sorted(weights)))
        defensive_flags.append(is_defensive(spec, mom, date))
        prev = weights

    returns = pd.Series(rets, index=pd.DatetimeIndex(dates), name=spec.name)
    equity = (1 + returns).cumprod() * 10_000.0
    return BacktestOutput(
        returns=returns,
        equity=equity,
        selections=pd.Series(picks, index=returns.index),
        defensive_ratio=float(np.mean(defensive_flags)) if defensive_flags else 0.0,
    )
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/taa/test_backtest.py -v`
Expected: PASS (6개)

- [ ] **Step 5: 커밋**

```bash
git add src/opt_portfolio/taa/backtest.py tests/taa/test_backtest.py
git commit -m "feat(taa): 월별 리밸런싱 엔진

신호는 t월말 종가로 정하고 수익은 t+1월에 얻는다. 같은 달 수익을 쓰면
룩어헤드이고 테스트로 고정했다.

비용은 회전한 만큼만 문다. 매달 물리면 정적 배분 기준선이 부당하게 불리해져
60/40 과의 비교가 망가진다 — 그 비교가 이 작업의 판정 기준이라 중요하다."
```

---

## Task 5: 사전 등록 9개 구성

**Files:**
- Create: `src/opt_portfolio/taa/registry.py`
- Test: `tests/taa/test_registry.py`

**Interfaces:**
- Consumes: Task 3 의 `StrategySpec`
- Produces: `REGISTERED: dict[str, StrategySpec]` — 정확히 9개. `ma_overlay: frozenset[str]`, `tranche: frozenset[str]` — 어느 구성이 어떤 변형을 쓰는지

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# tests/taa/test_registry.py
from __future__ import annotations

from opt_portfolio.taa.registry import MA_OVERLAY, REGISTERED, TRANCHE

EXPECTED = {
    "spy", "static_60_40", "vaa_g4", "baa_agg", "baa_bal",
    "baa_agg_ma", "baa_bal_ma", "baa_bal_tranche", "baa_bal_ma_tranche",
}


class TestRegistry:
    def test_exactly_nine_configurations(self) -> None:
        """DSR 의 n_trials 가 9 다. 늘리면 관문이 무의미해진다."""
        assert len(REGISTERED) == 9
        assert set(REGISTERED) == EXPECTED

    def test_vaa_uses_13612w_for_selection(self) -> None:
        assert REGISTERED["vaa_g4"].selection == "13612w"

    def test_baa_uses_sma13_for_selection(self) -> None:
        assert REGISTERED["baa_agg"].selection == "sma13"
        assert REGISTERED["baa_bal"].selection == "sma13"

    def test_vaa_canary_equals_its_offensive_universe(self) -> None:
        """VAA 의 병 — 경보기와 투자 대상이 같다."""
        spec = REGISTERED["vaa_g4"]
        assert set(spec.canary) == set(spec.offensive)

    def test_baa_canary_differs_from_offensive(self) -> None:
        """BAA 의 해법 — 분리한다."""
        spec = REGISTERED["baa_agg"]
        assert set(spec.canary) != set(spec.offensive)

    def test_baa_has_cash_dual_momentum_vaa_does_not(self) -> None:
        assert REGISTERED["baa_bal"].cash_ticker == "BIL"
        assert REGISTERED["vaa_g4"].cash_ticker is None

    def test_baa_balanced_holds_six_offensive_three_defensive(self) -> None:
        spec = REGISTERED["baa_bal"]
        assert spec.top_n_offensive == 6
        assert spec.top_n_defensive == 3

    def test_variant_flags_reference_registered_names(self) -> None:
        assert MA_OVERLAY <= set(REGISTERED)
        assert TRANCHE <= set(REGISTERED)

    def test_static_baseline_has_no_signals(self) -> None:
        assert REGISTERED["static_60_40"].static_weights == {"SPY": 0.6, "IEF": 0.4}
        assert REGISTERED["spy"].static_weights == {"SPY": 1.0}
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/taa/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: opt_portfolio.taa.registry`

- [ ] **Step 3: 최소 구현**

```python
# src/opt_portfolio/taa/registry.py
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
    "SPY", "QQQ", "IWM", "VGK", "EWJ", "EEM", "VNQ", "DBC", "GLD", "TLT", "HYG", "LQD",
)


def _spy() -> StrategySpec:
    return StrategySpec(
        name="spy", canary=(), offensive=(), defensive=(),
        top_n_offensive=0, top_n_defensive=0, static_weights={"SPY": 1.0},
    )


def _static_60_40() -> StrategySpec:
    return StrategySpec(
        name="static_60_40", canary=(), offensive=(), defensive=(),
        top_n_offensive=0, top_n_defensive=0, static_weights={"SPY": 0.6, "IEF": 0.4},
    )


def _vaa_g4() -> StrategySpec:
    # 경보기와 투자 대상이 같다 — 이것이 VAA 의 병이다.
    return StrategySpec(
        name="vaa_g4",
        canary=("SPY", "EFA", "EEM", "AGG"),
        offensive=("SPY", "EFA", "EEM", "AGG"),
        defensive=("LQD", "IEF", "SHY"),
        top_n_offensive=1, top_n_defensive=1,
        selection="13612w", cash_ticker=None,
    )


def _baa(name: str, offensive: tuple[str, ...], top_off: int, top_def: int) -> StrategySpec:
    return StrategySpec(
        name=name,
        canary=_BAA_CANARY,
        offensive=offensive,
        defensive=_BAA_DEFENSIVE,
        top_n_offensive=top_off, top_n_defensive=top_def,
        selection="sma13", cash_ticker="BIL",
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

#: DSR 에 넘길 시도 횟수. `len(REGISTERED)` 와 반드시 같아야 한다.
N_TRIALS: int = 9
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/taa/test_registry.py -v`
Expected: PASS (9개)

- [ ] **Step 5: 커밋**

```bash
git add src/opt_portfolio/taa/registry.py tests/taa/test_registry.py
git commit -m "feat(taa): 사전 등록 9개 구성 — 이것이 DSR 의 n_trials 다

결과를 보고 목록을 늘리면 DSR 관문이 무의미해진다. 테스트가 개수를 9로
고정한다.

VAA 는 canary 와 offensive 가 같고 BAA 는 다르다는 것을 테스트로 박아뒀다 —
그 차이가 이 작업 전체의 가설이기 때문이다."
```

---

## Task 6: 200일 이평 오버레이와 트랜치

**Files:**
- Modify: `src/opt_portfolio/taa/backtest.py`
- Test: `tests/taa/test_variants.py`

**Interfaces:**
- Consumes: Task 4 의 `run_backtest`
- Produces:
  - `run_with_ma_overlay(spec, daily, benchmark="SPY", ma_days=200, **kw) -> BacktestOutput`
  - `run_with_tranches(spec, daily, n_tranches=4, **kw) -> BacktestOutput`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# tests/taa/test_variants.py
from __future__ import annotations

import numpy as np
import pandas as pd

from opt_portfolio.taa.backtest import run_backtest, run_with_ma_overlay, run_with_tranches
from opt_portfolio.taa.strategy import StrategySpec

SPEC = StrategySpec(
    name="allspy", canary=(), offensive=(), defensive=(),
    top_n_offensive=0, top_n_defensive=0, static_weights={"SPY": 1.0},
)


def _crash_then_recover() -> pd.DataFrame:
    """앞 절반 상승, 뒤 절반 급락 — 이평 오버레이가 뒤쪽을 잘라야 한다."""
    n = 40 * 21
    up = np.linspace(100, 200, n // 2)
    down = np.linspace(200, 90, n - n // 2)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    px = np.concatenate([up, down])
    return pd.DataFrame({"SPY": px, "IEF": np.full(n, 100.0)}, index=idx)


class TestMaOverlay:
    def test_overlay_reduces_drawdown_in_a_crash(self) -> None:
        daily = _crash_then_recover()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        overlaid = run_with_ma_overlay(SPEC, daily, cost_bps=0.0)

        def mdd(eq: pd.Series) -> float:
            return float((eq / eq.cummax() - 1).min())

        assert mdd(overlaid.equity) > mdd(plain.equity)

    def test_overlay_is_flat_when_always_above_ma(self) -> None:
        n = 40 * 21
        idx = pd.date_range("2010-01-01", periods=n, freq="B")
        daily = pd.DataFrame(
            {"SPY": np.linspace(100, 300, n), "IEF": np.full(n, 100.0)}, index=idx
        )
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        overlaid = run_with_ma_overlay(SPEC, daily, cost_bps=0.0)

        pd.testing.assert_series_equal(plain.returns, overlaid.returns)


class TestTranches:
    def test_tranche_returns_have_lower_dispersion(self) -> None:
        """트랜치는 수익을 좇는 게 아니라 분산을 줄이는 장치다."""
        daily = _crash_then_recover()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        spread = run_with_tranches(SPEC, daily, cost_bps=0.0)

        assert spread.returns.std() <= plain.returns.std() * 1.01

    def test_tranche_output_has_same_index_as_plain(self) -> None:
        daily = _crash_then_recover()
        plain = run_backtest(SPEC, daily, cost_bps=0.0)
        spread = run_with_tranches(SPEC, daily, cost_bps=0.0)

        assert spread.returns.index.equals(plain.returns.index)
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/taa/test_variants.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_with_ma_overlay'`

- [ ] **Step 3: 구현을 덧붙인다**

`src/opt_portfolio/taa/backtest.py` 끝에 추가:

```python
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
    invested = (daily[benchmark] > ma).resample("ME").last().shift(1).fillna(True)

    aligned = invested.reindex(base.returns.index).fillna(True).astype(bool)
    returns = base.returns.where(aligned, 0.0)
    return BacktestOutput(
        returns=returns,
        equity=(1 + returns).cumprod() * 10_000.0,
        selections=base.selections.where(aligned, "CASH"),
        defensive_ratio=base.defensive_ratio,
    )


def run_with_tranches(
    spec: StrategySpec,
    daily: pd.DataFrame,
    n_tranches: int = 4,
    **kwargs: object,
) -> BacktestOutput:
    """자본을 `n_tranches` 로 나눠 서로 다른 주에 리밸런싱한 평균.

    단일 자산 + 월말 리밸런싱은 timing luck 에 취약하다 — 거래일 하루 차이로
    결과가 갈린다. **분산을 줄이는 장치이지 수익을 좇는 파라미터가 아니다.**
    """
    outs = []
    for offset in range(n_tranches):
        shifted = daily.shift(-offset * 5).dropna(how="all")
        outs.append(run_backtest(spec, shifted, **kwargs))  # type: ignore[arg-type]

    common = outs[0].returns.index
    for o in outs[1:]:
        common = common.intersection(o.returns.index)

    returns = sum(o.returns.reindex(common) for o in outs) / n_tranches
    assert isinstance(returns, pd.Series)
    returns.name = spec.name
    return BacktestOutput(
        returns=returns,
        equity=(1 + returns).cumprod() * 10_000.0,
        selections=outs[0].selections.reindex(common),
        defensive_ratio=float(np.mean([o.defensive_ratio for o in outs])),
    )
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/taa/test_variants.py -v`
Expected: PASS (4개)

- [ ] **Step 5: 커밋**

```bash
git add src/opt_portfolio/taa/backtest.py tests/taa/test_variants.py
git commit -m "feat(taa): 200일 이평 오버레이와 리밸런싱 트랜치

개선안 둘. 근거가 우리 실측이거나 문서화된 약점인 것만 넣었다.

이평 오버레이는 팩터 엔진에서 MDD 를 -63.8% -> -23.7% 로 줄인 장치다. 다만
BAA 카나리아가 이미 추세를 판정하므로 여기서는 효과가 없거나 마이너스일 수
있다 — 그 예상을 코드 주석과 설계 문서 §7 에 미리 적었다.

트랜치는 timing luck 완화용이다. 분산을 줄이는 장치라 수익을 좇는 파라미터가
아니고, 그래서 과최적화 위험이 낮다."
```

---

## Task 7: 평가 파이프라인

**Files:**
- Create: `scripts/run_taa.py`
- Test: `tests/taa/test_evaluate.py`
- Create: `src/opt_portfolio/taa/evaluate.py`

**Interfaces:**
- Consumes: Task 1~6 전부, `factor.research.overfitting`, `analysis.metrics`
- Produces:
  - `evaluate_all(daily, start, end, cost_bps) -> tuple[pd.DataFrame, pd.DataFrame]` — (지표표, 수익률 행렬)
  - `verdict(metrics: pd.DataFrame, pbo: float) -> pd.DataFrame` — 채택 기준 적용 결과

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# tests/taa/test_evaluate.py
from __future__ import annotations

import numpy as np
import pandas as pd

from opt_portfolio.taa.evaluate import ADOPTION, summarize, verdict


def _returns(mu: float, sd: float, n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2007-06-30", periods=n, freq="ME")
    return pd.Series(rng.normal(mu, sd, n), index=idx)


class TestSummarize:
    def test_reports_cagr_mdd_calmar(self) -> None:
        row = summarize("x", _returns(0.008, 0.03))

        assert {"cagr", "mdd", "calmar", "sharpe", "vol"} <= set(row)
        assert row["calmar"] == row["cagr"] / abs(row["mdd"])

    def test_annualizes_monthly_with_twelve(self) -> None:
        """월별인데 252 로 연율화하면 이 저장소가 세 번째로 같은 실수를 한다."""
        r = _returns(0.008, 0.03)
        row = summarize("x", r)

        assert row["vol"] == float(r.std() * np.sqrt(12))


class TestVerdict:
    def test_rejects_when_drawdown_exceeds_limit(self) -> None:
        m = pd.DataFrame(
            [{"name": "a", "mdd": -0.35, "calmar": 2.0, "dsr": 0.99}]
        ).set_index("name")
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "MDD" in out.loc["a", "reason"]

    def test_rejects_when_dsr_below_gate(self) -> None:
        m = pd.DataFrame(
            [{"name": "a", "mdd": -0.15, "calmar": 2.0, "dsr": 0.80}]
        ).set_index("name")
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "DSR" in out.loc["a", "reason"]

    def test_rejects_when_calmar_below_baseline(self) -> None:
        m = pd.DataFrame(
            [{"name": "a", "mdd": -0.15, "calmar": 0.4, "dsr": 0.99}]
        ).set_index("name")
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "60/40" in out.loc["a", "reason"]

    def test_adopts_only_when_all_gates_pass(self) -> None:
        m = pd.DataFrame(
            [{"name": "a", "mdd": -0.15, "calmar": 0.9, "dsr": 0.99}]
        ).set_index("name")
        out = verdict(m, pbo=0.1, baseline_calmar=0.5)

        assert out.loc["a", "adopted"]

    def test_high_pbo_rejects_everything(self) -> None:
        """PBO 가 주 관문이다 — 넘으면 개별 성적과 무관하게 전부 기각."""
        m = pd.DataFrame(
            [{"name": "a", "mdd": -0.10, "calmar": 3.0, "dsr": 1.0}]
        ).set_index("name")
        out = verdict(m, pbo=0.6, baseline_calmar=0.5)

        assert not out.loc["a", "adopted"]
        assert "PBO" in out.loc["a", "reason"]

    def test_adoption_constants_match_the_spec(self) -> None:
        assert ADOPTION["mdd_limit"] == -0.20
        assert ADOPTION["dsr_gate"] == 0.95
        assert ADOPTION["pbo_limit"] == 0.5
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/taa/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: opt_portfolio.taa.evaluate`

- [ ] **Step 3: 최소 구현**

```python
# src/opt_portfolio/taa/evaluate.py
"""9개 구성 평가와 채택 판정.

**PBO 가 주 관문이다.** 여기서의 탐색은 파라미터 적합이 아니라 *9개 구성 중
하나 고르기* 이고, CSCV 가 정확히 그 상황을 다룬다 — 인샘플 1등이 아웃샘플에서
중앙값 아래로 떨어질 확률.

DSR 은 보조다. **월별 수익률을 그대로 넘긴다 — 연율화 금지** (docstring 요구).
이 저장소는 연율화 주기를 이미 두 번 틀렸다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..factor.research.overfitting import deflated_sharpe_ratio
from .registry import N_TRIALS

#: 설계 문서 §6 의 채택 기준. 결과를 보고 바꾸지 않는다.
ADOPTION: dict[str, float] = {
    "mdd_limit": -0.20,
    "dsr_gate": 0.95,
    "pbo_limit": 0.5,
}

_MONTHS_PER_YEAR = 12


def summarize(name: str, returns: pd.Series) -> dict[str, float | str]:
    """월별 수익률 → 지표. 연율화는 12 로 한다."""
    equity = (1 + returns).cumprod()
    years = len(returns) / _MONTHS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    mdd = float((equity / equity.cummax() - 1).min())
    vol = float(returns.std() * np.sqrt(_MONTHS_PER_YEAR))
    return {
        "name": name,
        "cagr": cagr,
        "mdd": mdd,
        "vol": vol,
        "calmar": cagr / abs(mdd) if mdd else float("nan"),
        "sharpe": float(returns.mean() * _MONTHS_PER_YEAR / vol) if vol else float("nan"),
        "dsr": float(deflated_sharpe_ratio(returns, n_trials=N_TRIALS)),
        "months": float(len(returns)),
    }


def verdict(metrics: pd.DataFrame, pbo: float, baseline_calmar: float) -> pd.DataFrame:
    """채택 기준 적용. 하나라도 못 넘으면 기각하고 이유를 적는다."""
    rows = []
    for name, row in metrics.iterrows():
        reasons: list[str] = []
        if pbo > ADOPTION["pbo_limit"]:
            reasons.append(f"PBO {pbo:.2f} > {ADOPTION['pbo_limit']}")
        if row["mdd"] < ADOPTION["mdd_limit"]:
            reasons.append(f"MDD {row['mdd']:.1%} 가 한도 초과")
        if row["dsr"] < ADOPTION["dsr_gate"]:
            reasons.append(f"DSR {row['dsr']:.3f} < {ADOPTION['dsr_gate']}")
        if row["calmar"] <= baseline_calmar:
            reasons.append(f"Calmar {row['calmar']:.2f} 가 60/40({baseline_calmar:.2f}) 이하")
        rows.append({"name": name, "adopted": not reasons, "reason": " · ".join(reasons) or "—"})
    return pd.DataFrame(rows).set_index("name")
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/taa/test_evaluate.py -v`
Expected: PASS (8개)

- [ ] **Step 5: 실행 스크립트를 만든다**

```python
# scripts/run_taa.py
"""사전 등록 9개 구성을 돌리고 PBO/DSR 로 판정한다.

    uv run python scripts/run_taa.py

결과가 나쁘면 목록을 늘리고 싶어진다. **늘리지 않는다** — 그 순간 DSR 이
의미를 잃는다.
"""

from __future__ import annotations

import pandas as pd

from opt_portfolio.factor.research.overfitting import probability_of_backtest_overfitting
from opt_portfolio.taa.backtest import run_backtest, run_with_ma_overlay, run_with_tranches
from opt_portfolio.taa.data import load_prices
from opt_portfolio.taa.evaluate import summarize, verdict
from opt_portfolio.taa.registry import MA_OVERLAY, REGISTERED, TRANCHE

START, END = pd.Timestamp("2007-06-30"), pd.Timestamp("2026-08-31")
COST_BPS = 10.0


def main() -> int:
    tickers = sorted({t for spec in REGISTERED.values() for t in spec.tickers()})
    daily = load_prices(tickers)
    print(f"가격 패널: {daily.shape[1]}종목 {daily.index.min().date()} ~ {daily.index.max().date()}")

    rets: dict[str, pd.Series] = {}
    rows = []
    for name, spec in REGISTERED.items():
        kw = {"start": START, "end": END, "cost_bps": COST_BPS}
        if name in TRANCHE:
            out = run_with_tranches(spec, daily, **kw)
        elif name in MA_OVERLAY:
            out = run_with_ma_overlay(spec, daily, **kw)
        else:
            out = run_backtest(spec, daily, **kw)
        rets[name] = out.returns
        rows.append({**summarize(name, out.returns), "defensive": out.defensive_ratio})

    metrics = pd.DataFrame(rows).set_index("name")
    matrix = pd.DataFrame(rets).dropna(how="any")
    pbo = probability_of_backtest_overfitting(matrix).pbo

    print("\n=== 지표 ===")
    print(metrics.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nPBO = {pbo:.3f}  (관측 {len(matrix)}개월 × 구성 {matrix.shape[1]}개)")
    print("\n=== 판정 ===")
    print(verdict(metrics, pbo, float(metrics.loc["static_60_40", "calmar"])).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 실제로 돌린다**

Run: `uv run python scripts/run_taa.py`
Expected: 9행 지표표 + PBO + 판정표. **어느 구성이든 `months` 가 200 미만이면 멈추고 데이터 구간을 확인한다.**

- [ ] **Step 7: 커밋**

```bash
git add src/opt_portfolio/taa/evaluate.py scripts/run_taa.py tests/taa/test_evaluate.py
git commit -m "feat(taa): 평가 파이프라인 — PBO 를 주 관문으로

여기서의 탐색은 파라미터 적합이 아니라 9개 구성 중 하나 고르기다. CSCV 가
정확히 그 상황을 다루므로 PBO 가 주 관문이고 DSR 이 보조다.

채택 기준을 상수로 박고 테스트로 고정했다 — 결과를 보고 기준을 바꾸는 것을
막기 위해서다. 기각 사유를 문자열로 남겨 왜 떨어졌는지가 표에 보인다.

월별 연율화에 12 를 쓰는 것도 테스트로 고정했다. 이 저장소는 연율화 주기를
이미 두 번 틀렸다."
```

---

## Task 8: 결과 기록과 문서 정리

**Files:**
- Modify: `docs/factor-system/07-experiment-log.md` (또는 신규 `docs/taa/01-results.md`)
- Modify: `CLAUDE.md`
- Modify: `README.md`, `README.ko.md`

- [ ] **Step 1: `CLAUDE.md` 에 격리 규칙 예외를 기록한다**

`## 절대 규칙` 아래 서브시스템 설명에 추가:

```markdown
**세 번째 서브시스템 `taa/` 가 있다** (2026-08-17). ETF 전술적 배분이며
`factor/` 도 `strategies/` 도 아니다. **격리 규칙의 예외를 하나 열었다** —
`taa/` 는 `factor.research.overfitting`(DSR·PBO)을 import 한다. 그 함수들이
평범한 수익률 시계열만 받아 결합이 얇고, 관문 없이 만든 성과를 믿지 않는 것이
이 저장소의 규약이기 때문이다. 다른 방향의 import 는 금지한다.
```

- [ ] **Step 2: 결과를 문서에 적는다**

`scripts/run_taa.py` 출력을 그대로 옮기고, **설계 문서 §7 의 예상과 대조해서** 맞았는지 틀렸는지를 적는다. 특히:

- BAA > VAA 였는가
- 200일 이평은 효과가 없거나 마이너스였는가 (예상 2)
- 60/40 을 이긴 구성이 있는가 (예상 5 — 없으면 전부 기각)

**한계를 결과와 같은 자리에 적는다** — 230개월, 하락 3회.

- [ ] **Step 3: README 의 VAA 섹션을 고친다**

현재 README 는 VAA 최종 자산을 `$29,000` 으로 적고 있는데 실측은 `$22,813` 이다. 그리고 Sharpe 는 연율화 버그로 20.9배 부풀어 있었다. 실측값으로 교체하고, **왜 6% 인가**(방어 55% · SHY 를 연 0.05% 로 24%)를 함께 적는다.

- [ ] **Step 4: 전체 검증**

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/opt_portfolio/taa/
uv run pre-commit run --all-files
```

- [ ] **Step 5: 커밋**

```bash
git add -A
git commit -m "docs(taa): 9개 구성 결과와 판정

설계 문서 §7 의 예상과 대조해 적었다. 맞은 것과 틀린 것을 함께 남기는 것이
다음번에 같은 실험을 두 번 하지 않는 방법이다.

한계를 결과 뒤 각주가 아니라 같은 자리에 적었다 — 230개월에 하락 3회이므로
'붕괴 방어가 작동한다' 는 주장은 사건 3개에 근거한다.

README 의 VAA 수치도 실측으로 교체했다. 최종 자산이 \$29,000 으로 적혀
있었으나 실제는 \$22,813 이고, Sharpe 는 연율화 버그로 20.9배 부풀어 있었다."
```

---

## 자체 검토 결과

**스펙 커버리지** — 설계 문서 8개 절 전부 대응 확인:

| 스펙 절 | 대응 |
|---|---|
| §2.1 아키텍처 | Task 1~7 (파일 구조), Task 8 Step 1 (격리 예외 기록) |
| §2.2 데이터 · 배당조정 | Task 1 (`closeadj` 상수 + 테스트) |
| §3 기준선 | Task 5 (`spy`, `static_60_40`, `vaa_g4`, `baa_*`) |
| §4 사전 등록 9개 | Task 5 (`N_TRIALS=9` + 개수 고정 테스트) |
| §5 PBO 주관문 · DSR | Task 7 |
| §6 채택 기준 | Task 7 (`ADOPTION` 상수 + 테스트) |
| §7 결과 전 예상 | Task 8 Step 2 (예상과 대조) |
| §8 한계 | Task 8 Step 2 |

**미대응 하나** — 스펙 §5 의 *timing luck 측정(리밸런싱일 1~21일)* 과 *비용 민감도 30bps* 는 채택된 구성이 나온 뒤 하는 강건성 검사다. 채택 없음으로 끝나면 불필요하므로 **Task 8 이후 후속으로 남긴다.** 스펙이 이를 "시도 횟수에 넣지 않는다"고 명시했으므로 관문에 영향이 없다.

**타입 일관성** — `StrategySpec` 필드명이 Task 3 정의와 Task 5·6·7 사용에서 일치. `BacktestOutput` 필드(`returns`·`equity`·`selections`·`defensive_ratio`)가 Task 4 정의와 Task 6·7 사용에서 일치. `summarize` 가 내는 키(`cagr`·`mdd`·`calmar`·`dsr`)가 `verdict` 가 읽는 키와 일치.
