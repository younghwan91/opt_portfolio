"""
백테스트 티어시트 — 자기완결 HTML 리포트

UX 원칙: **이 리포트의 임무는 자기기만 방지다.**
- 최상단은 수익률이 아니라 판정 게이트 5개 (OOS Sharpe → DSR → 안정성 →
  PBO → 비용 민감도). walk-forward 를 안 거쳤으면 게이트가 '미실행'으로
  비어 있고, 리포트 전체에 '참고용' 배지가 붙는다.
- 모든 실행이 불변 HTML 파일 하나로 남는다 — 외부 의존성 없음(CDN·폰트·
  이미지 전부 인라인), 나중에 열어도 그대로. 연구 증거는 세션이 아니라
  파일이어야 한다.

시각 규격은 dataviz 검증 팔레트를 따른다 (라이트/다크 자동, 색약 안전).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from opt_portfolio.factor.backtest.engine import BacktestResult
from opt_portfolio.factor.optimize.walkforward import WalkForwardResult

# ------------------------------------------------------------------ 팔레트
# dataviz 레퍼런스 팔레트 (검증 완료: CVD ΔE·대비 게이트 통과)

_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --neg: #d03b3b;
  --good: #0ca30c; --warn: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --good-text: #006300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --neg: #e66767;
    --good-text: #0ca30c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --neg: #e66767;
  --good-text: #0ca30c;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px; line-height: 1.5;
}
main { max-width: 1080px; margin: 0 auto; padding: 24px 20px 48px; }
h1 { font-size: 20px; margin: 0; }
h2 { font-size: 15px; margin: 28px 0 10px; color: var(--ink-2); }
.sub { color: var(--muted); font-size: 12.5px; margin-top: 4px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 99px;
  font-size: 12px; font-weight: 600; vertical-align: middle; margin-left: 10px; }
.badge.ref { background: var(--warn); color: #0b0b0b; }
.badge.official { background: var(--good); color: #fff; }
.card { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: 16px; }
.row { display: grid; gap: 10px; }
.gates { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.tiles { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
.gate .name { font-size: 12px; color: var(--ink-2); }
.gate .verdict { font-weight: 700; margin-top: 2px; }
.gate .detail { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
.tile .label { font-size: 12px; color: var(--ink-2); }
.tile .value { font-size: 22px; font-weight: 650; margin-top: 2px; }
.tile .value.pos { color: var(--good-text); }
.tile .value.negv { color: var(--neg); }
svg text { font-family: inherit; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: right; color: var(--ink-2); font-weight: 600;
  border-bottom: 1px solid var(--axis); padding: 6px 10px; }
th:first-child, td:first-child { text-align: left; }
td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; }
.footer { margin-top: 32px; color: var(--muted); font-size: 12px;
  border-top: 1px solid var(--grid); padding-top: 12px; }
#tip { position: fixed; pointer-events: none; background: var(--surface);
  border: 1px solid var(--ring); border-radius: 6px; padding: 6px 9px;
  font-size: 12px; display: none; box-shadow: 0 2px 8px rgba(0,0,0,.12); z-index: 9; }
"""

_TOOLTIP_JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => {
    tip.innerHTML = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 180) + 'px';
    tip.style.top = (e.clientY + 14) + 'px';
  });
  el.addEventListener('mouseleave', () => tip.style.display = 'none');
});
document.querySelectorAll('svg.hoverline').forEach(svg => {
  const pts = JSON.parse(svg.dataset.points);
  const cross = svg.querySelector('.cross');
  svg.addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * svg.viewBox.baseVal.width;
    let best = 0, bd = 1e18;
    for (let i = 0; i < pts.length; i++) {
      const d = Math.abs(pts[i][0] - x);
      if (d < bd) { bd = d; best = i; }
    }
    const p = pts[best];
    cross.setAttribute('transform', `translate(${p[0]},0)`);
    cross.style.display = 'block';
    cross.querySelector('circle').setAttribute('cy', p[1]);
    tip.innerHTML = p[2]; tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 180) + 'px';
    tip.style.top = (e.clientY + 14) + 'px';
  });
  svg.addEventListener('mouseleave', () => {
    cross.style.display = 'none'; tip.style.display = 'none';
  });
});
"""


@dataclass(frozen=True)
class Gate:
    """판정 게이트 하나 — 05-math-spec.md 의 5단계."""

    name: str
    verdict: str  # "통과" | "실패" | "미실행"
    detail: str
    color: str  # good | critical | muted


def _judgment_gates(wf: WalkForwardResult | None) -> list[Gate]:
    if wf is None:
        skip = "walk-forward(optimize) 실행 필요"
        return [
            Gate("① OOS Sharpe > 0.5", "미실행", skip, "muted"),
            Gate("② Deflated Sharpe > 0.95", "미실행", skip, "muted"),
            Gate("③ 파라미터 안정성", "미실행", skip, "muted"),
            Gate("④ PBO < 0.3", "미실행", skip, "muted"),
            Gate("⑤ 비용 2배 민감도", "미실행", skip, "muted"),
        ]
    oos_sharpe = wf.sharpe()
    dsr = wf.deflated_sharpe()
    stability = wf.param_stability()
    # 안정성: 수치형 파라미터의 폴드 간 IQR / 전체 범위 ≤ 1/3
    numeric = stability.select_dtypes("number")
    if len(numeric.columns) and len(numeric) > 1:
        spread = (
            (numeric.quantile(0.75) - numeric.quantile(0.25))
            / (numeric.max() - numeric.min()).replace(0, np.nan)
        ).max()
        stable = bool(spread <= 1 / 3) if np.isfinite(spread) else True
        stab_detail = f"최대 IQR 비율 {spread:.2f}" if np.isfinite(spread) else "단일값"
    else:
        stable, stab_detail = True, "수치 파라미터 없음"
    return [
        Gate(
            "① OOS Sharpe > 0.5",
            "통과" if oos_sharpe > 0.5 else "실패",
            f"OOS Sharpe {oos_sharpe:.2f}",
            "good" if oos_sharpe > 0.5 else "critical",
        ),
        Gate(
            "② Deflated Sharpe > 0.95",
            "통과" if dsr > 0.95 else "실패",
            f"DSR {dsr:.3f} (시도 {wf.n_trials_total}회 반영)",
            "good" if dsr > 0.95 else "critical",
        ),
        Gate(
            "③ 파라미터 안정성",
            "통과" if stable else "실패",
            stab_detail,
            "good" if stable else "critical",
        ),
        Gate("④ PBO < 0.3", "미실행", "시도 로그로 별도 계산", "muted"),
        Gate("⑤ 비용 2배 민감도", "미실행", "비용 2배 재실행으로 확인", "muted"),
    ]


# ------------------------------------------------------------------ SVG 조각


def _scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi <= lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def _equity_svg(equity: pd.Series, width: int = 1040, height: int = 260) -> str:
    """로그 스케일 에쿼티 곡선 + 크로스헤어 툴팁."""
    eq = equity.dropna()
    log_eq = np.log(eq.to_numpy())
    lo, hi = float(log_eq.min()), float(log_eq.max())
    pad_l, pad_r, pad_t, pad_b = 46, 14, 10, 22

    xs = np.linspace(pad_l, width - pad_r, len(eq))
    ys = np.array([_scale(v, lo, hi, height - pad_b, pad_t) for v in log_eq])
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    # 그리드: 25/50/100% 수익 수준선
    grid_rows = []
    for mult in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        lv = np.log(mult)
        if lo - 1e-9 <= lv <= hi + 1e-9:
            y = _scale(lv, lo, hi, height - pad_b, pad_t)
            grid_rows.append(
                f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"'
                f' stroke="var(--grid)" stroke-width="1"/>'
                f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end"'
                f' fill="var(--muted)" font-size="11">{mult:.2g}×</text>'
            )
    # x축: 연도 눈금
    years = pd.Series(range(len(eq)), index=eq.index).resample("YS").first().dropna()
    for pos_idx in years:
        x = xs[int(pos_idx)]
        label = eq.index[int(pos_idx)].year
        grid_rows.append(
            f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle"'
            f' fill="var(--muted)" font-size="11">{label}</text>'
        )

    points = [
        [round(float(x), 1), round(float(y), 1), f"{d.date()}<br><b>{v:.3f}×</b>"]
        for x, y, d, v in zip(xs, ys, eq.index, eq.to_numpy())
    ]
    step = max(1, len(points) // 400)  # 툴팁 데이터 경량화
    return f"""
<svg class="hoverline" viewBox="0 0 {width} {height}" data-points='{json.dumps(points[::step])}'
     style="width:100%;height:auto;display:block">
  {"".join(grid_rows)}
  <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}"
        stroke="var(--axis)" stroke-width="1"/>
  <polyline points="{line}" fill="none" stroke="var(--s1)" stroke-width="2"
            stroke-linejoin="round"/>
  <g class="cross" style="display:none">
    <line y1="{pad_t}" y2="{height - pad_b}" stroke="var(--axis)" stroke-width="1"
          stroke-dasharray="3,3"/>
    <circle r="4" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/>
  </g>
</svg>"""


def _drawdown_svg(equity: pd.Series, width: int = 1040, height: int = 130) -> str:
    eq = equity.dropna()
    dd = (eq / eq.cummax() - 1.0).to_numpy()
    lo = float(dd.min())
    pad_l, pad_r, pad_t, pad_b = 46, 14, 6, 8

    xs = np.linspace(pad_l, width - pad_r, len(dd))
    ys = np.array([_scale(v, lo, 0.0, height - pad_b, pad_t) for v in dd])
    top = _scale(0.0, lo, 0.0, height - pad_b, pad_t)
    area = (
        f"M {xs[0]:.1f},{top:.1f} "
        + " ".join(f"L {x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        + f" L {xs[-1]:.1f},{top:.1f} Z"
    )
    label_y = _scale(lo, lo, 0.0, height - pad_b, pad_t)
    return f"""
<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block">
  <line x1="{pad_l}" y1="{top:.1f}" x2="{width - pad_r}" y2="{top:.1f}"
        stroke="var(--axis)" stroke-width="1"/>
  <path d="{area}" fill="var(--neg)" opacity="0.28"/>
  <text x="{pad_l - 6}" y="{label_y + 4:.1f}" text-anchor="end" fill="var(--muted)"
        font-size="11">{lo:.0%}</text>
</svg>"""


def _diverging_color(v: float, vmax: float) -> str:
    """월수익 → blue(+) / red(−), 회색 중립점. 팔레트 diverging 규약."""
    if not np.isfinite(v) or vmax <= 0:
        return "var(--grid)"
    t = min(abs(v) / vmax, 1.0)
    # 중립 #f0efec ↔ 파랑 #2a78d6 / 빨강 #d03b3b 선형 보간 (라이트 기준 —
    # 다크에서도 판독 가능하도록 알파 없이 고정 헥스 사용)
    neutral = (240, 239, 236)
    pole = (42, 120, 214) if v >= 0 else (208, 59, 59)
    rgb = tuple(round(n + (p - n) * t) for n, p in zip(neutral, pole))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _monthly_heatmap(returns: pd.Series) -> str:
    monthly = (1.0 + returns.dropna()).resample("ME").prod() - 1.0
    if monthly.empty:
        return "<p class='sub'>데이터 없음</p>"
    frame = monthly.to_frame("r")
    frame["y"], frame["m"] = frame.index.year, frame.index.month
    pivot = frame.pivot_table(index="y", columns="m", values="r")
    vmax = float(np.nanquantile(np.abs(pivot.to_numpy()), 0.95)) or 0.05

    head = "".join(f"<th>{m}월</th>" for m in range(1, 13)) + "<th>연간</th>"
    body = []
    for year, row in pivot.iterrows():
        cells = []
        for m in range(1, 13):
            v = row.get(m, np.nan)
            if np.isfinite(v):
                cells.append(
                    f'<td data-tip="{year}-{m:02d}<br><b>{v:+.2%}</b>"'
                    f' style="background:{_diverging_color(v, vmax)};'
                    f'color:#0b0b0b">{v * 100:+.1f}</td>'
                )
            else:
                cells.append("<td></td>")
        annual = float((1 + row.dropna()).prod() - 1)
        cls = "pos" if annual >= 0 else "negv"
        cells.append(f'<td style="font-weight:650" class="{cls}">{annual:+.1%}</td>')
        body.append(f"<tr><td>{year}</td>{''.join(cells)}</tr>")
    return (
        "<table><thead><tr><th></th>" + head + "</tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table>"
        "<p class='sub'>단위 %. 파랑 = 상승, 빨강 = 하락 (셀에 마우스를 올리면 정확한 값)</p>"
    )


# ------------------------------------------------------------------ 렌더링


def _fmt(v: float, kind: str) -> str:
    if not np.isfinite(v):
        return "—"
    if kind == "pct":
        return f"{v:+.1%}"
    if kind == "ratio":
        return f"{v:.2f}"
    return f"{v:,.0f}"


def render_tearsheet(
    result: BacktestResult,
    *,
    title: str,
    config_summary: dict[str, str],
    ic_table: pd.DataFrame | None = None,
    walk_forward: WalkForwardResult | None = None,
    coverage: pd.DataFrame | None = None,
) -> str:
    """티어시트 HTML 문자열을 만든다 — 파일 저장은 호출 측."""
    stats = result.stats()
    official = walk_forward is not None
    badge = (
        '<span class="badge official">공식 — walk-forward OOS</span>'
        if official
        else '<span class="badge ref">참고용 — 단일 백테스트</span>'
    )

    gates_html = "".join(
        f"""<div class="card gate">
  <div class="name">{g.name}</div>
  <div class="verdict" style="color:var(--{g.color})">{g.verdict}</div>
  <div class="detail">{g.detail}</div>
</div>"""
        for g in _judgment_gates(walk_forward)
    )

    tile_defs = [
        ("CAGR", stats.get("cagr", np.nan), "pct"),
        ("Sharpe", stats.get("sharpe", np.nan), "ratio"),
        ("Sortino", stats.get("sortino", np.nan), "ratio"),
        ("최대 낙폭", stats.get("max_drawdown", np.nan), "pct"),
        ("Calmar", stats.get("calmar", np.nan), "ratio"),
        ("평균 회전율", stats.get("avg_turnover", np.nan), "ratio"),
        ("리밸런싱", stats.get("n_rebalances", np.nan), "int"),
    ]
    tiles_html = "".join(
        f"""<div class="card tile"><div class="label">{label}</div>
<div class="value {("negv" if kind == "pct" and v < 0 else "")}">{_fmt(v, kind)}</div></div>"""
        for label, v, kind in tile_defs
    )

    config_html = " · ".join(f"{k} {v}" for k, v in config_summary.items())

    ic_html = ""
    if ic_table is not None and not ic_table.empty:
        rows = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in ic_table.itertuples(index=False)
        )
        heads = "".join(f"<th>{c}</th>" for c in ic_table.columns)
        ic_html = (
            "<h2>팩터 예측력 (Rank IC, 월간)</h2><div class='card'>"
            f"<table><thead><tr>{heads}</tr></thead><tbody>{rows}</tbody></table>"
            "<p class='sub'>t ≥ 2 가 유의 기준. 소표본에서는 과신 금지.</p></div>"
        )

    coverage_html = ""
    if coverage is not None:
        coverage_html = " · ".join(
            f"{r.table} {r.rows:,}행" for r in coverage.itertuples() if r.rows
        )

    warning = (
        ""
        if official
        else """<p class="sub" style="color:var(--serious);font-weight:600">
⚠️ 이 결과는 파라미터 선택 과정을 검증하지 않았다. 전략 승인 판단에 사용 금지 —
공식 성과는 opt-factor optimize (walk-forward + DSR) 로만 만든다.</p>"""
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    span = f"{result.returns.index[0].date()} ~ {result.returns.index[-1].date()}"

    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
<main>
  <h1>{title}{badge}</h1>
  <div class="sub">{span} · {config_html}</div>
  {warning}

  <h2>판정 게이트 — 이 5개를 통과해야 전략이다</h2>
  <div class="row gates">{gates_html}</div>

  <h2>성과 요약</h2>
  <div class="row tiles">{tiles_html}</div>

  <h2>누적 수익 (로그 스케일)</h2>
  <div class="card">{_equity_svg(result.equity)}</div>

  <h2>드로다운</h2>
  <div class="card">{_drawdown_svg(result.equity)}</div>

  <h2>월별 수익률</h2>
  <div class="card" style="overflow-x:auto">{_monthly_heatmap(result.returns)}</div>

  {ic_html}

  <div class="footer">
    생성 {generated} · 데이터 {coverage_html or "—"} ·
    opt_portfolio factor engine
  </div>
</main>
<div id="tip"></div>
<script>{_TOOLTIP_JS}</script>
"""
