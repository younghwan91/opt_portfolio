#!/usr/bin/env python3
"""
README 성과 차트를 만든다 — light/dark 각각.

세 장을 만드는 이유가 각각 다르다. 형태는 데이터가 답해야 할 질문이 정한다.

1. `performance`  — "23.6년 동안 무슨 일이 있었나" → 시간축 선그래프(로그 스케일).
   누적 성장은 배수로 읽어야 하므로 선형 축이면 초반이 뭉개진다.
2. `risk-return`  — "오늘 잰 것들이 위험 대비 어디에 있나" → 산점도.
   CAGR 하나만 보면 초소형주가 이겨 보이고, 낙폭을 같이 놓아야 그림이 뒤집힌다.
   점이 아홉이라 색으로 구분하지 않고 **직접 라벨 + 상태색 강조**를 쓴다
   (`dataviz` 규칙: all-pairs 형태는 범주 색 3개가 상한이다).
3. `guards`       — "방어 장치를 켜면 어떻게 되나" → 같은 전략의 before/after 선그래프.
   숫자로 "Sharpe 1.047 → −0.224" 라고 적는 것보다 곡선이 꺾이는 걸 보는 편이 빠르다.

팔레트는 `dataviz` 스킬의 검증된 기본값이고 `validate_palette.js` 로 통과를
확인했다 — light/dark 3슬롯 모두 PASS, aqua 의 대비 WARN 은 직접 라벨로 해소한다.

    uv run python scripts/make_readme_charts.py
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path.home() / "data"
OUT = Path(__file__).resolve().parents[1] / "docs/images"

#: dataviz 기본 팔레트. 왼쪽이 light, 오른쪽이 dark — 자동 반전이 아니라
#: 어두운 표면에 맞춰 따로 고른 단계다.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "critical": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#33322f",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "critical": "#e66767",
    },
}


def load_daily(name: str) -> pd.Series:
    """walk-forward 산출물(일별 수익률) 하나를 읽는다."""
    for cand in (DATA / f"{name}.json", DATA / "results" / f"{name}.json"):
        if cand.exists():
            raw = json.loads(cand.read_text())
            idx = pd.DatetimeIndex([pd.Timestamp(int(k), unit="s") for k in raw])
            return pd.Series(list(raw.values()), index=idx).sort_index()
    raise FileNotFoundError(f"산출물이 없다: {name}")


def _style(ax: plt.Axes, t: Mapping[str, Any]) -> None:
    """축·격자를 뒤로 물린다 — 데이터가 앞에 오게."""
    ax.set_facecolor(str(t["surface"]))
    ax.grid(True, color=str(t["grid"]), linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(str(t["grid"]))
    ax.tick_params(colors=str(t["muted"]), labelsize=9)


def _save(fig: plt.Figure, name: str, mode: str, t: Mapping[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{mode}.png"
    fig.savefig(path, dpi=160, facecolor=str(t["surface"]), bbox_inches="tight")
    plt.close(fig)
    return path


def chart_performance(mode: str) -> Path:
    """E안 누적 성장 vs SPY vs 60/40 — 로그 스케일."""
    t = THEME[mode]
    e15, e50 = load_daily("oos_lean_timed_train5"), load_daily("oos_lean_timed_slip50")
    spy = _spy_daily(e15.index)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    _style(ax, t)
    series = [
        ("대형주 E안 (15bps)", (1 + e15).cumprod(), t["series"][0], 2.2),
        ("E안 (슬립 50bps)", (1 + e50).cumprod(), t["series"][2], 1.6),
        ("SPY 바이앤홀드", (1 + spy).cumprod(), t["series"][1], 1.6),
    ]
    for label, eq, color, lw in series:
        ax.plot(eq.index, eq.values, color=color, linewidth=lw, label=label)
        # 직접 라벨 — 범례에만 기대지 않는다(대비 WARN 슬롯의 구제 규칙)
        ax.annotate(
            f"{eq.iloc[-1]:.0f}배",
            (eq.index[-1], eq.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold",
        )
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 40])
    ax.set_yticklabels(["1배", "2배", "5배", "10배", "20배", "40배"])
    ax.set_title(
        "대형주 5팩터 + 200일 타이밍 — walk-forward 검증 구간 (2002-12 ~ 2026-08)",
        color=str(t["ink"]),
        fontsize=11.5,
        pad=12,
        loc="left",
    )
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(str(t["muted"]))
    return _save(fig, "performance", mode, t)


def _spy_daily(index: pd.DatetimeIndex) -> pd.Series:
    """벤치마크 일별 수익률 — 스토어의 SPY 를 검증 구간에 맞춘다."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"attach '{DATA / 'us_micro.duckdb'}' as s (read_only)")
    df = con.execute("select date, close from s.prices where ticker='SPY' order by date").df()
    px = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"])).astype(float)
    return px.pct_change(fill_method=None).reindex(index).fillna(0.0)


def chart_risk_return(mode: str) -> Path:
    """오늘 잰 전부를 위험·수익 평면에 놓는다."""
    t = THEME[mode]
    pts = [
        ("E안 대형주", 16.34, -24.3, True),
        ("E안 50bps", 14.91, -24.3, True),
        ("초소형주 (방어 OFF)", 23.74, -23.7, False),
        ("초소형주 (방어 ON)", -16.42, -96.2, False),
        ("SPY", 12.45, -41.8, False),
        ("60/40", 8.86, -25.1, False),
        ("VAA-G4", 6.07, -20.9, False),
        ("BAA-agg", 8.82, -16.5, False),
        ("BAA-bal +MA", 6.72, -8.3, False),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax, t)
    for label, cagr, mdd, adopted in pts:
        color = str(t["series"][0]) if adopted else str(t["muted"])
        if cagr < 0:
            color = str(t["critical"])
        ax.scatter(
            abs(mdd),
            cagr,
            s=150 if adopted else 80,
            color=color,
            zorder=3,
            edgecolors=str(t["surface"]),
            linewidths=2,
        )
        ax.annotate(
            label,
            (abs(mdd), cagr),
            xytext=(9, 4),
            textcoords="offset points",
            color=color if adopted or cagr < 0 else str(t["muted"]),
            fontsize=9,
            fontweight="bold" if adopted else "normal",
        )
    ax.axhline(0, color=str(t["grid"]), linewidth=1)
    ax.set_xlabel("최대낙폭 (%, 오른쪽일수록 위험)", color=str(t["muted"]), fontsize=9.5)
    ax.set_ylabel("연평균 수익률 (%)", color=str(t["muted"]), fontsize=9.5)
    ax.set_title(
        "위험 대비 수익 — 왼쪽 위가 좋다  (2026-08 측정, 구간은 각 전략의 검증창)",
        color=str(t["ink"]),
        fontsize=11.5,
        pad=12,
        loc="left",
    )
    return _save(fig, "risk-return", mode, t)


def chart_guards(mode: str) -> Path:
    """같은 전략, 방어 장치만 켜고 끈 결과."""
    t = THEME[mode]
    off, on = load_daily("results/oos_quantus_train5"), load_daily("oos_cost_guards")
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    _style(ax, t)
    for label, s, color, lw in [
        ("방어 장치 OFF — 표제였던 값", off, t["series"][0], 2.2),
        # `$` 는 matplotlib 에서 mathtext 구간을 연다 — 이스케이프하지 않으면
        # "$5 · 거래대금 $1M" 사이가 수식으로 해석되어 한글이 두부(□)로 깨진다.
        ("방어 장치 ON — 슬립 50bps · 최소주가 \\$5 · 거래대금 \\$1M", on, t["critical"], 2.2),
    ]:
        eq = (1 + s).cumprod()
        ax.plot(eq.index, eq.values, color=color, linewidth=lw, label=label)
        ax.annotate(
            f"{eq.iloc[-1]:.1f}배" if eq.iloc[-1] >= 1 else f"{eq.iloc[-1]:.3f}배",
            (eq.index[-1], eq.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold",
        )
    ax.set_yscale("log")
    # 기본 로그 눈금은 `10^2` 로 나온다 — 다른 차트가 "배" 로 읽히는데 여기만
    # 지수 표기면 같은 축을 두 언어로 읽게 된다.
    ax.set_yticks([0.04, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.04배", "0.1배", "1배", "10배", "100배"])
    ax.minorticks_off()
    ax.set_title(
        "초소형주 전략 — 설계 문서가 필수라 적은 방어 장치를 켜면",
        color=str(t["ink"]),
        fontsize=11.5,
        pad=12,
        loc="left",
    )
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(str(t["muted"]))
    return _save(fig, "guards", mode, t)


def main() -> int:
    font = _pick_korean_font()
    plt.rcParams["font.family"] = font
    # mathtext 는 본문 폰트를 따라가지 않는다 — 지정하지 않으면 축 라벨의
    # 마이너스 기호 등에서 `Font 'rm' does not have a glyph` 로 두부가 난다.
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = font
    plt.rcParams["axes.unicode_minus"] = False
    made = []
    for mode in ("light", "dark"):
        made += [chart_performance(mode), chart_risk_return(mode), chart_guards(mode)]
    for p in made:
        rel = p.relative_to(Path(__file__).resolve().parents[1])
        print(f"  {rel}  {p.stat().st_size // 1024}KB")
    return 0


def _pick_korean_font() -> str:
    """한글이 깨지면 차트가 무용지물이라, 등록 후 실제 렌더링까지 확인한다.

    두 가지 함정이 있다.

    1. `findSystemFonts()` 전수 스캔은 손상된 폰트 파일 하나에 걸려 죽는다
       (`RuntimeError: Can not load face`). 후보 경로만 직접 확인한다.
    2. `NotoSansCJK-Regular.ttc` 는 컬렉션이라 matplotlib 에 **`Noto Sans CJK JP`**
       라는 이름으로 등록된다. 이름은 JP 지만 같은 파일에 한글 글리프가 있다.
       파일 경로로 골라놓고 이름을 `KR` 로 넘기면 조용히 폴백되어 라벨이
       두부(□)가 된다 — 실제로 한 번 그렇게 나왔다.
    """
    from matplotlib import font_manager

    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
    ):
        if not Path(path).exists():
            continue
        font_manager.fontManager.addfont(path)
        name = font_manager.FontProperties(fname=path).get_name()
        if _renders_hangul(name):
            return name

    raise SystemExit(
        "한글을 렌더링할 폰트가 없다 — 차트 라벨이 두부(□)로 깨진다. "
        "`sudo apt install fonts-nanum` 후 다시 실행하라."
    )


def _renders_hangul(font_name: str) -> bool:
    """이름만 믿지 않고 글리프가 실제로 있는지 본다."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=font_name), fallback_to_default=False
        )
        return FT2Font(path).get_char_index(ord("대")) != 0
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
