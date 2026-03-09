"""
Streamlit Web UI for Portfolio Management

This module provides an interactive web interface for the portfolio
management system with extended features and professional visualizations.

퀀트 관점:
- 시각화는 의사결정 품질 향상의 핵심
- 실시간 모니터링으로 빠른 대응 가능
- 인터랙티브 분석으로 전략 이해도 향상
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from opt_portfolio.analysis.backtest import BacktestEngine  # noqa: E402
from opt_portfolio.analysis.data_fetcher import DataFetcher  # noqa: E402
from opt_portfolio.analysis.optimizer import PortfolioOptimizer  # noqa: E402
from opt_portfolio.analysis.performance import PerformanceAnalyzer  # noqa: E402
from opt_portfolio.analysis.risk import RiskAnalyzer  # noqa: E402
from opt_portfolio.config import ASSETS, UI  # noqa: E402
from opt_portfolio.core.cache import get_cache  # noqa: E402
from opt_portfolio.core.portfolio import Portfolio  # noqa: E402
from opt_portfolio.strategies.momentum import MomentumAnalyzer  # noqa: E402
from opt_portfolio.strategies.ou_process import OUForecaster  # noqa: E402
from opt_portfolio.strategies.vaa import VAAStrategy  # noqa: E402

PAGES = [
    "📊 VAA 분석",
    "💼 포트폴리오 관리",
    "📈 전략 백테스트",
    "📉 리스크 분석",
    "🔬 포트폴리오 최적화",
    "🎓 전략 교육",
]


# ─────────────────────────────────────────────────────────────
# Session state & sidebar
# ─────────────────────────────────────────────────────────────


def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        "selected_etf": None,
        "vaa_result": None,
        "portfolio": None,
        "backtest_results": None,
        "win_probs": None,
        "forecast_df": None,
        "momentum_df": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar() -> str:
    """Render sidebar with navigation, freshness indicator, and strategy summary."""
    st.sidebar.title("🚀 포트폴리오 관리")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("페이지 이동", PAGES, index=0)

    # ── Data freshness indicator (ui-freshness) ──────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 데이터 상태")
    cache = get_cache()
    try:
        stats = cache.get_cache_stats()
        if not stats.empty:
            st.sidebar.metric("캐시된 티커 수", len(stats))
            # Check cache file modification time as freshness proxy
            cache_dir = Path.home() / ".opt_portfolio"
            db_files = list(cache_dir.glob("*.db")) + list(cache_dir.glob("*.duckdb"))
            if db_files:
                import time

                last_mod = max(f.stat().st_mtime for f in db_files)
                age_hours = (time.time() - last_mod) / 3600
                if age_hours < 1:
                    st.sidebar.success(f"✅ 최신 ({age_hours * 60:.0f}분 전)")
                elif age_hours < 24:
                    st.sidebar.info(f"🕐 {age_hours:.1f}시간 전 업데이트")
                else:
                    st.sidebar.warning(f"⚠️ {age_hours / 24:.1f}일 전 (갱신 권장)")
            if st.sidebar.button("🔄 캐시 갱신", use_container_width=True):
                with st.spinner("캐시 갱신 중..."):
                    try:
                        cache.optimize()
                        st.sidebar.success("갱신 완료!")
                    except Exception as e:
                        st.sidebar.error(f"갱신 실패: {e}")
        else:
            st.sidebar.info("캐시 데이터 없음")
    except Exception:
        st.sidebar.info("캐시 상태 확인 불가")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 VAA 전략 요약")
    st.sidebar.markdown(
        """
    **비중 배분:**
    - 50% → VAA 선택 ETF
    - 12.5%씩 → SPY, TLT, GLD, BIL

    **리밸런싱:** 월 1회
    """
    )

    return page


# ─────────────────────────────────────────────────────────────
# VAA Analysis page
# ─────────────────────────────────────────────────────────────


def render_vaa_page():
    """Render VAA analysis page with OU fan chart and custom weights."""
    st.header("📊 VAA ETF 선택 분석")
    st.markdown("*고급 예측 기반 Vigilant Asset Allocation*")

    col1, col2, col3 = st.columns(3)
    with col1:
        analysis_date = st.date_input("분석 기준일", value=date.today(), max_value=date.today())
    with col2:
        _strategy = st.selectbox(  # future use: switch analysis mode
            "선택 전략",
            ["현재 모멘텀(VAA)", "1개월 예측", "3개월 예측", "모멘텀 변화(Δ)"],
        )
    with col3:
        show_forecast = st.checkbox("승률(Win Prob.) 보기", value=True)

    # ── Advanced: custom momentum weights (ui-momentum-weights) ──
    with st.expander("⚙️ 고급: 모멘텀 가중치 커스텀 (Keller 기본 12:4:2:1)"):
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        with wcol1:
            w1m = st.slider("1개월 가중치", 1, 20, 12)
        with wcol2:
            w3m = st.slider("3개월 가중치", 1, 10, 4)
        with wcol3:
            w6m = st.slider("6개월 가중치", 1, 5, 2)
        with wcol4:
            w12m = st.slider("12개월 가중치", 1, 3, 1)
        custom_weights = [w1m, w3m, w6m, w12m]
        st.caption(
            f"현재 설정: {w1m}:{w3m}:{w6m}:{w12m} "
            f"({'Keller 원본' if custom_weights == [12, 4, 2, 1] else '커스텀'})"
        )

    if st.button("🔍 VAA 분석 실행", type="primary", use_container_width=True):
        with st.spinner("OU 예측 기반 시장 분석 중..."):
            try:
                vaa = VAAStrategy(use_forecasting=True)
                result = vaa.select(analysis_date)
                st.session_state.vaa_result = result
                st.session_state.selected_etf = result.selected_etf

                if show_forecast:
                    win_probs, forecast_df = vaa.get_win_probabilities(analysis_date)
                    st.session_state.win_probs = win_probs
                    st.session_state.forecast_df = forecast_df

                # Pre-load momentum data for fan chart
                ma = MomentumAnalyzer(use_cache=True)
                end_d = date.today()
                start_d = end_d - timedelta(days=2 * 365)
                tickers = list(ASSETS.AGGRESSIVE_TICKERS)
                mom_df = ma.calculate_historical_momentum(tickers, start_d, end_d)
                st.session_state.momentum_df = mom_df

            except Exception as e:
                st.error(f"분석 실패: {e}")
                return

    if not st.session_state.vaa_result:
        return

    result = st.session_state.vaa_result

    # ── Hero banner: today's pick ────────────────────────────
    if result.is_defensive:
        st.warning(f"🛡️ **방어 모드** — 선택 ETF: **{result.selected_etf}**")
    else:
        st.success(f"📈 **성장 모드** — 선택 ETF: **{result.selected_etf}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔥 공격 자산군")
        if not result.aggressive_ranking.empty:
            df = result.aggressive_ranking.copy()
            df["상태"] = df["Momentum Score"].apply(lambda x: "🟢" if x > 0 else "🔴")
            st.dataframe(df.round(2), use_container_width=True)
    with col2:
        st.markdown("#### 🛡️ 방어 자산군")
        if not result.protective_ranking.empty:
            df = result.protective_ranking.copy()
            df["상태"] = df["Momentum Score"].apply(lambda x: "🟢" if x > 0 else "🔴")
            st.dataframe(df.round(2), use_container_width=True)

    if result.strategy_recommendations:
        st.markdown("### 📊 전략별 추천")
        rec_data = [
            {"전략": s, "추천 자산": d.get("asset", ""), "점수": round(d.get("score", 0), 2)}
            for s, d in result.strategy_recommendations.items()
            if "asset" in d
        ]
        if rec_data:
            st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

    if st.session_state.win_probs is not None and not st.session_state.win_probs.empty:
        st.markdown("### 🎲 승률 (다음달 기준)")
        fig = px.bar(
            x=st.session_state.win_probs.values * 100,
            y=st.session_state.win_probs.index,
            orientation="h",
            labels={"x": "승률(%)", "y": "자산"},
            title="최고 성과 자산 확률",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── OU Fan Chart (ui-ou-chart) ───────────────────────────
    if st.session_state.momentum_df is not None and not st.session_state.momentum_df.empty:
        render_ou_fan_chart(st.session_state.momentum_df)

    render_momentum_chart()


def render_ou_fan_chart(momentum_df: pd.DataFrame) -> None:
    """Render OU process forecast fan chart (ui-ou-chart)."""
    st.markdown("### 🔮 OU 프로세스 예측 Fan Chart")

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        ticker = st.selectbox("예측 자산", options=momentum_df.columns.tolist(), key="ou_ticker")

    series = momentum_df[ticker].dropna()
    if len(series) < 15:
        st.info("데이터 부족 (최소 15개 필요)")
        return

    forecaster = OUForecaster(num_simulations=500)

    # Confidence interval table
    ci_rows = []
    for months in [1, 3, 6]:
        try:
            lower, mean, upper = forecaster.get_confidence_interval(
                series, months=months, confidence=0.90
            )
            ci_rows.append(
                {
                    "예측 기간": f"{months}개월 후",
                    "하단(5%)": round(lower, 1),
                    "중앙값": round(mean, 1),
                    "상단(95%)": round(upper, 1),
                    "방어 전환 위험": "🔴 높음"
                    if upper < 0
                    else ("🟡 주의" if lower < 0 else "🟢 낮음"),
                }
            )
        except Exception:
            pass

    with col_info:
        if ci_rows:
            st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

    # Fan chart using Monte Carlo paths
    try:
        mean_val, std_val, paths = forecaster.simulate(series, months=6, return_paths=True)

        history = series.tail(18)
        last_date = pd.Timestamp(history.index[-1])
        future_dates = pd.date_range(start=last_date, periods=7, freq="ME")[1:]
        current_val = float(history.iloc[-1])

        fig = go.Figure()

        # Historical
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history.values,
                mode="lines",
                name="실제 모멘텀",
                line=dict(color="royalblue", width=2),
            )
        )

        # Fan from Monte Carlo paths
        if paths is not None and paths.ndim == 2:
            p10 = np.percentile(paths, 10, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p50 = np.percentile(paths, 50, axis=0)
            p75 = np.percentile(paths, 75, axis=0)
            p90 = np.percentile(paths, 90, axis=0)

            x_all = [last_date] + list(future_dates)

            # 80% band (p10–p90)
            fig.add_trace(
                go.Scatter(
                    x=x_all,
                    y=[current_val] + list(p90),
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="p90",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_all,
                    y=[current_val] + list(p10),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(100,149,237,0.18)",
                    name="80% 신뢰구간",
                )
            )
            # 50% band (p25–p75)
            fig.add_trace(
                go.Scatter(
                    x=x_all,
                    y=[current_val] + list(p75),
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="p75",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_all,
                    y=[current_val] + list(p25),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(100,149,237,0.35)",
                    name="50% 신뢰구간",
                )
            )
            # Median forecast
            fig.add_trace(
                go.Scatter(
                    x=x_all,
                    y=[current_val] + list(p50),
                    mode="lines",
                    name="예측 중앙값",
                    line=dict(color="crimson", width=2, dash="dash"),
                )
            )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="0선 (방어 전환)")
        fig.update_layout(
            title=f"{ticker} 모멘텀 예측 (6개월, 500회 시뮬레이션)",
            xaxis_title="날짜",
            yaxis_title="모멘텀 점수",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.info(f"Fan chart 생성 중 오류 (데이터 부족 가능): {e}")


def render_momentum_chart():
    """Render historical momentum chart."""
    st.markdown("### 📈 모멘텀 히스토리")

    with st.expander("히스토리 차트 보기", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            years = st.slider("조회 기간(년)", 1, 5, 2, key="mom_hist_years")
        with col2:
            asset_type = st.radio("자산군", ["공격", "방어"], key="mom_hist_type")

        tickers = (
            list(ASSETS.AGGRESSIVE_TICKERS)
            if asset_type == "공격"
            else list(ASSETS.PROTECTIVE_TICKERS)
        )
        ma = MomentumAnalyzer()
        end_d = date.today()
        start_d = end_d - timedelta(days=years * 365)
        mom_df = ma.calculate_historical_momentum(tickers, start_d, end_d)

        if not mom_df.empty:
            fig = go.Figure()
            for tkr in mom_df.columns:
                fig.add_trace(go.Scatter(x=mom_df.index, y=mom_df[tkr], mode="lines", name=tkr))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title=f"{asset_type} 자산군 모멘텀 히스토리",
                xaxis_title="날짜",
                yaxis_title="모멘텀 점수",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Portfolio Management page
# ─────────────────────────────────────────────────────────────


def render_portfolio_page():
    """Render portfolio management page with multi-scenario comparison."""
    st.header("💼 포트폴리오 관리")

    if not st.session_state.selected_etf:
        st.warning("⚠️ 먼저 VAA 분석을 실행해 ETF를 선택하세요.")
        return

    selected_etf = st.session_state.selected_etf
    st.success(f"🎯 VAA 선택 ETF: **{selected_etf}** (목표 비중 50%)")

    st.markdown("### 📝 현재 보유 종목 입력")
    col1, col2 = st.columns(2)
    holdings = {}

    with col1:
        st.markdown(f"**{selected_etf}** (목표: 50%)")
        holdings[selected_etf] = st.number_input(
            f"{selected_etf} 주식 수", min_value=0, value=0, step=1
        )
        st.markdown("**SPY** (목표: 12.5%)")
        holdings["SPY"] = st.number_input("SPY 주식 수", min_value=0, value=0, step=1)
        st.markdown("**TLT** (목표: 12.5%)")
        holdings["TLT"] = st.number_input("TLT 주식 수", min_value=0, value=0, step=1)

    with col2:
        st.markdown("**GLD** (목표: 12.5%)")
        holdings["GLD"] = st.number_input("GLD 주식 수", min_value=0, value=0, step=1)
        st.markdown("**BIL** (목표: 12.5%)")
        holdings["BIL"] = st.number_input("BIL 주식 수", min_value=0, value=0, step=1)
        st.markdown("**추가 투자금**")
        additional_cash = st.number_input("추가 투자금 ($)", min_value=0.0, value=0.0, step=100.0)

    portfolio = Portfolio.from_dict(holdings)
    portfolio.update_prices()

    if any(s > 0 for s in holdings.values()):
        st.markdown("### 📊 현재 배분 현황")
        allocation = portfolio.get_allocation()
        if not allocation.empty:
            fig = px.pie(
                allocation.reset_index(),
                values="Value",
                names="Ticker",
                title="현재 포트폴리오 배분",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(allocation.round(2), use_container_width=True)
            st.metric("💰 총 포트폴리오 가치", f"${portfolio.total_value:,.2f}")

    st.markdown("### ⚖️ 리밸런싱 계산")
    if st.button("⚡ 최적 리밸런싱 계산", type="primary", use_container_width=True):
        with st.spinner("최적 거래 계산 중..."):
            recs = portfolio.calculate_rebalance(selected_etf, additional_cash)
            if "error" in recs:
                st.error(recs["error"])
            else:
                render_rebalancing_results(recs)

    # ── Multi-scenario comparison (ui-multi-scenario) ────────
    with st.expander("🔁 시나리오 비교 (추가 현금별 리밸런싱)"):
        st.markdown("세 가지 추가 현금 시나리오를 동시에 비교합니다.")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            cash_a = st.number_input(
                "시나리오 A ($)", min_value=0.0, value=1000.0, step=500.0, key="scen_a"
            )
        with s_col2:
            cash_b = st.number_input(
                "시나리오 B ($)", min_value=0.0, value=5000.0, step=500.0, key="scen_b"
            )
        with s_col3:
            cash_c = st.number_input(
                "시나리오 C ($)", min_value=0.0, value=10000.0, step=500.0, key="scen_c"
            )

        if st.button("📊 시나리오 비교 실행", use_container_width=True):
            with st.spinner("3개 시나리오 계산 중..."):
                scenarios = {"A": cash_a, "B": cash_b, "C": cash_c}
                s_col1, s_col2, s_col3 = st.columns(3)

                for col, (label, cash) in zip([s_col1, s_col2, s_col3], scenarios.items()):
                    with col:
                        st.markdown(f"**시나리오 {label}: +${cash:,.0f}**")
                        try:
                            p = Portfolio.from_dict(holdings)
                            p.update_prices()
                            r = p.calculate_rebalance(selected_etf, cash)
                            if "error" not in r:
                                st.metric("목표 가치", f"${r['total_target_value']:,.0f}")
                                # Show allocation errors summary
                                errors = r.get("allocation_errors", {})
                                avg_err = (
                                    sum(abs(e["percentage_error"]) for e in errors.values())
                                    / len(errors)
                                    if errors
                                    else 0
                                )
                                status = "🟢" if avg_err < 1 else ("🟡" if avg_err < 2.5 else "🔴")
                                st.markdown(f"평균 오차: {status} **{avg_err:.2f}%**")
                                if r.get("transactions"):
                                    buy_count = sum(
                                        1
                                        for t in r["transactions"].values()
                                        if t["action"] == "BUY"
                                    )
                                    sell_count = sum(
                                        1
                                        for t in r["transactions"].values()
                                        if t["action"] == "SELL"
                                    )
                                    st.markdown(f"매수 {buy_count}건 / 매도 {sell_count}건")
                                else:
                                    st.success("거래 불필요")
                            else:
                                st.error(r["error"])
                        except Exception as e:
                            st.error(f"계산 오류: {e}")


def render_rebalancing_results(recommendations: dict):
    """Render rebalancing recommendations."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("현재 가치", f"${recommendations['current_value']:,.2f}")
    with col2:
        st.metric("추가 투자금", f"${recommendations['additional_cash']:,.2f}")
    with col3:
        st.metric("목표 가치", f"${recommendations['total_target_value']:,.2f}")
    with col4:
        st.metric("최종 가치", f"${recommendations['final_portfolio_value']:,.2f}")

    if recommendations.get("transactions"):
        st.markdown("### 📋 필요 거래 내역")
        col1, col2 = st.columns(2)
        sells, buys = [], []
        for ticker, trans in recommendations["transactions"].items():
            row = {
                "ETF": ticker,
                "주식 수": trans["shares"],
                "가격": f"${trans['price']:.2f}",
            }
            if trans["action"] == "SELL":
                row["매도금액"] = f"${trans.get('proceeds', 0):,.2f}"
                sells.append(row)
            else:
                row["매수금액"] = f"${trans.get('cost', 0):,.2f}"
                buys.append(row)

        with col1:
            if sells:
                st.markdown("**🔴 매도**")
                st.dataframe(pd.DataFrame(sells), use_container_width=True, hide_index=True)
        with col2:
            if buys:
                st.markdown("**🟢 매수**")
                st.dataframe(pd.DataFrame(buys), use_container_width=True, hide_index=True)
    else:
        st.success("✅ 포트폴리오 최적화 완료 — 추가 거래 불필요")

    st.markdown("### 🎯 최종 배분 분석")
    error_data = [
        {
            "ETF": tkr,
            "주식 수": recommendations["optimized_portfolio"].get(tkr, 0),
            "목표 %": f"{e['target_percentage']:.1f}%",
            "실제 %": f"{e['actual_percentage']:.1f}%",
            "오차": f"{e['percentage_error']:+.1f}%",
            "상태": "🟢"
            if abs(e["percentage_error"]) < 1
            else ("🟡" if abs(e["percentage_error"]) < 3 else "🔴"),
        }
        for tkr, e in recommendations["allocation_errors"].items()
    ]
    st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)

    total_error = sum(
        abs(e["percentage_error"]) for e in recommendations["allocation_errors"].values()
    )
    avg_error = total_error / max(len(recommendations["allocation_errors"]), 1)
    if avg_error < 1:
        st.success(f"🟢 배분 품질 우수 (평균 오차: {avg_error:.2f}%)")
    elif avg_error < 2.5:
        st.info(f"🟡 배분 품질 양호 (평균 오차: {avg_error:.2f}%)")
    else:
        st.warning(f"🟠 추가 자본 투입으로 오차 개선 가능 (평균 오차: {avg_error:.2f}%)")


# ─────────────────────────────────────────────────────────────
# Backtest page
# ─────────────────────────────────────────────────────────────


def render_backtest_page():
    """Render backtesting page with Export functionality."""
    st.header("📈 전략 백테스트")
    st.markdown("*VAA 전략을 과거 데이터로 비교 분석*")

    col1, col2 = st.columns(2)
    with col1:
        years = st.slider("백테스트 기간(년)", 5, 20, 15)
    with col2:
        strategies = st.multiselect(
            "테스트할 전략 선택",
            ["현재", "1개월 예측", "3개월 예측", "6개월 예측", "모멘텀 변화"],
            default=["현재", "1개월 예측", "모멘텀 변화"],
        )

    if st.button("🚀 백테스트 실행", type="primary", use_container_width=True):
        with st.spinner(f"{years}년 백테스트 실행 중..."):
            try:
                engine = BacktestEngine()
                results = engine.run_vaa_backtest(years=years, strategies=strategies)
                if results:
                    st.session_state.backtest_results = results
            except Exception as e:
                st.error(f"백테스트 실패: {e}")

    if not st.session_state.backtest_results:
        return

    results = st.session_state.backtest_results

    # Summary table
    st.markdown("### 📊 성과 요약")
    summary_data = [
        {
            "전략": name,
            "최종 가치": f"${r.final_capital:,.0f}",
            "CAGR": f"{r.cagr:.1%}",
            "샤프지수": f"{r.sharpe_ratio:.2f}",
            "최대 낙폭": f"{r.max_drawdown:.1%}",
            "승률": f"{r.win_rate:.1%}",
        }
        for name, r in results.items()
    ]
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Equity curves
    st.markdown("### 📈 자본곡선 비교")
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(
            go.Scatter(x=r.equity_curve.index, y=r.equity_curve.values, mode="lines", name=name)
        )
    fig.update_layout(
        title="전략별 자본곡선",
        xaxis_title="날짜",
        yaxis_title="포트폴리오 가치($)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    st.markdown("### 📉 낙폭 분석")
    fig2 = go.Figure()
    for name, r in results.items():
        running_max = r.equity_curve.expanding().max()
        drawdown = (r.equity_curve - running_max) / running_max * 100
        fig2.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=name,
                fill="tozeroy",
            )
        )
    fig2.update_layout(title="전략별 낙폭", xaxis_title="날짜", yaxis_title="낙폭(%)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Export (ui-export) ───────────────────────────────────
    st.markdown("### 📤 결과 내보내기")
    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        # Summary CSV
        csv_summary = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 성과 요약 CSV 다운로드",
            data=csv_summary,
            file_name="backtest_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with ex_col2:
        # Monthly returns CSV
        monthly_data = {}
        for name, r in results.items():
            monthly_returns = r.equity_curve.pct_change().dropna()
            monthly_data[name] = monthly_returns
        if monthly_data:
            monthly_df = pd.DataFrame(monthly_data)
            monthly_df.index.name = "Date"
            csv_monthly = monthly_df.to_csv().encode("utf-8")
            st.download_button(
                "📥 월별 수익률 CSV 다운로드",
                data=csv_monthly,
                file_name="backtest_monthly_returns.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────
# Risk Dashboard page (ui-risk-page)
# ─────────────────────────────────────────────────────────────


def render_risk_page():
    """Render comprehensive risk analysis dashboard."""
    st.header("📉 리스크 분석 대시보드")
    st.markdown("*백테스트 결과 기반 정량 리스크 분석*")

    if not st.session_state.backtest_results:
        st.info("💡 먼저 **전략 백테스트** 페이지에서 백테스트를 실행하세요.")
        years = st.slider("빠른 백테스트 기간(년)", 5, 20, 10, key="risk_years")
        if st.button("🚀 리스크 분석용 백테스트 실행", type="primary", use_container_width=True):
            with st.spinner("백테스트 실행 중..."):
                try:
                    engine = BacktestEngine()
                    results = engine.run_vaa_backtest(years=years, strategies=["현재"])
                    st.session_state.backtest_results = results
                    st.rerun()
                except Exception as e:
                    st.error(f"백테스트 실패: {e}")
        return

    results = st.session_state.backtest_results
    strategy_name = st.selectbox("분석할 전략", list(results.keys()), key="risk_strategy")
    result = results[strategy_name]

    equity_curve = result.equity_curve
    returns = equity_curve.pct_change().dropna()

    # Get SPY as benchmark
    try:
        cache = get_cache()
        end_d = equity_curve.index[-1]
        start_d = equity_curve.index[0]
        spy_prices = cache.get_incremental_data(["SPY"], start_d, end_d)
        if not spy_prices.empty:
            spy_prices = spy_prices["SPY"].resample("ME").last()
            benchmark_returns = spy_prices.pct_change().dropna()
            benchmark_returns = benchmark_returns.reindex(returns.index).dropna()
            common_idx = returns.index.intersection(benchmark_returns.index)
            r_aligned = returns.loc[common_idx]
            b_aligned = benchmark_returns.loc[common_idx]
        else:
            r_aligned = returns
            b_aligned = None
    except Exception:
        r_aligned = returns
        b_aligned = None

    # Calculate all risk metrics
    risk_analyzer = RiskAnalyzer()
    perf_analyzer = PerformanceAnalyzer()

    try:
        risk_metrics = risk_analyzer.calculate_all_metrics(
            returns=r_aligned,
            prices=equity_curve,
            benchmark_returns=b_aligned,
        )

        # ── Key metrics cards ────────────────────────────────
        st.markdown("### 📊 핵심 리스크 지표")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("연간 변동성", f"{risk_metrics.volatility:.1%}")
        with m2:
            st.metric("샤프 비율", f"{risk_metrics.sharpe_ratio:.2f}")
        with m3:
            st.metric("소르티노 비율", f"{risk_metrics.sortino_ratio:.2f}")
        with m4:
            st.metric("VaR 95%", f"{risk_metrics.var_95:.1%}")
        with m5:
            st.metric("CVaR 95%", f"{risk_metrics.cvar_95:.1%}")
        with m6:
            beta_val = risk_metrics.beta
            st.metric("Beta vs SPY", f"{beta_val:.2f}" if beta_val is not None else "N/A")

        m7, m8 = st.columns([1, 5])
        with m7:
            st.metric("Calmar 비율", f"{risk_metrics.calmar_ratio:.2f}")

    except Exception as e:
        st.warning(f"리스크 지표 계산 중 오류: {e}")

    # ── Year-by-year returns (ui-risk-page) ─────────────────
    st.markdown("### 📅 연도별 성과")
    try:
        yearly_df = perf_analyzer.analyze_by_year(returns)
        if not yearly_df.empty:
            # Bar chart: annual returns
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in yearly_df["Return"]]
            fig = go.Figure(
                go.Bar(
                    x=yearly_df["Year"].astype(str),
                    y=(yearly_df["Return"] * 100).round(1),
                    marker_color=colors,
                    text=(yearly_df["Return"] * 100).round(1).astype(str) + "%",
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="연도별 수익률",
                xaxis_title="연도",
                yaxis_title="수익률(%)",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            yearly_display = yearly_df.copy()
            for col in ["Return", "Volatility"]:
                if col in yearly_display.columns:
                    yearly_display[col] = yearly_display[col].apply(lambda x: f"{x:.1%}")
            if "Sharpe" in yearly_display.columns:
                yearly_display["Sharpe"] = yearly_display["Sharpe"].apply(lambda x: f"{x:.2f}")
            st.dataframe(yearly_display, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"연도별 분석 오류: {e}")

    # ── Top 5 Drawdowns ──────────────────────────────────────
    st.markdown("### 🚨 주요 낙폭 구간 (Top 5)")
    try:
        dd_df = perf_analyzer.calculate_drawdown_analysis(equity_curve, top_n=5)
        if not dd_df.empty:
            dd_display = dd_df.copy()
            if "Max Drawdown" in dd_display.columns:
                dd_display["Max Drawdown"] = dd_display["Max Drawdown"].apply(lambda x: f"{x:.1%}")
            st.dataframe(dd_display, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"낙폭 분석 오류: {e}")

    # ── Market regime analysis ───────────────────────────────
    if b_aligned is not None:
        st.markdown("### 🌊 시장 레짐별 성과 (vs SPY)")
        try:
            regime = perf_analyzer.analyze_by_market_regime(r_aligned, b_aligned)
            r_cols = st.columns(3)
            labels = {"up_market": "📈 상승장", "down_market": "📉 하락장", "overall": "📊 전체"}
            for col, (key, label) in zip(r_cols, labels.items()):
                with col:
                    data = regime.get(key, {})
                    st.markdown(f"**{label}**")
                    if isinstance(data, dict):
                        for metric, val in list(data.items())[:4]:
                            if isinstance(val, float):
                                st.markdown(f"- {metric}: **{val:.2%}**")
        except Exception as e:
            st.info(f"레짐 분석 오류: {e}")

    # Return distribution histogram
    st.markdown("### 📊 월별 수익률 분포")
    try:
        fig = px.histogram(
            x=returns * 100,
            nbins=30,
            labels={"x": "월별 수익률(%)"},
            title="월별 수익률 분포",
            color_discrete_sequence=["steelblue"],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(
            x=float(returns.mean() * 100),
            line_dash="dot",
            line_color="green",
            annotation_text=f"평균 {returns.mean() * 100:.1f}%",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"분포 차트 오류: {e}")


# ─────────────────────────────────────────────────────────────
# Portfolio Optimizer page (ui-optimizer-page)
# ─────────────────────────────────────────────────────────────


def render_optimizer_page():
    """Render portfolio optimizer page with Efficient Frontier."""
    st.header("🔬 포트폴리오 최적화")
    st.markdown("*Sharpe Ratio 최대화 기반 자산 배분 최적화*")

    with st.expander("💡 최적화 원리", expanded=False):
        st.markdown(
            """
        포트폴리오 최적화는 VAA 선택 ETF와 4개 핵심 자산(SPY, TLT, GLD, BIL)의
        비중을 Grid Search로 탐색해 **Sharpe Ratio가 가장 높은 조합**을 찾습니다.

        - 탐색 범위: 각 자산 비중 20%~60%, 5% 단위
        - 기준 지표: Sharpe Ratio (수익률 / 변동성)
        - 제약 조건: 전체 비중 합 = 100%
        """
        )

    col1, col2 = st.columns(2)
    with col1:
        opt_years = st.slider("최적화 기간(년)", 3, 15, 10, key="opt_years")
    with col2:
        st.markdown("")
        st.markdown("")
        run_opt = st.button("🚀 최적화 실행", type="primary", use_container_width=True)

    if run_opt:
        with st.spinner(f"{opt_years}년 데이터로 최적 비중 탐색 중... (수십 초 소요)"):
            try:
                cache = get_cache()
                fetcher = DataFetcher(cache=cache)
                vaa_returns, core_returns = fetcher.get_component_returns(years=opt_years)

                if vaa_returns.empty or core_returns.empty:
                    st.error("데이터 수집 실패 — 캐시를 갱신하거나 네트워크를 확인하세요.")
                    return

                optimizer = PortfolioOptimizer()
                opt_result = optimizer.optimize(vaa_returns, core_returns)
                st.session_state.opt_result = opt_result
            except Exception as e:
                st.error(f"최적화 실패: {e}")
                return

    if not hasattr(st.session_state, "opt_result") or st.session_state.opt_result is None:
        return

    opt = st.session_state.opt_result

    # ── Best result cards ────────────────────────────────────
    st.markdown("### 🏆 최적화 결과")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("최적 Sharpe", f"{opt.best_sharpe:.3f}")
    with c2:
        st.metric("기대 CAGR", f"{opt.best_return:.1f}%")
    with c3:
        st.metric("기대 변동성", f"{opt.best_volatility:.1f}%")
    with c4:
        st.metric("기대 MDD", f"{opt.best_max_drawdown:.1f}%")

    # Best weights comparison
    st.markdown("### ⚖️ 최적 비중 vs 기본 비중")
    default_weights = {"VAA": 50.0, "SPY": 12.5, "TLT": 12.5, "GLD": 12.5, "BIL": 12.5}
    weight_df = pd.DataFrame(
        {
            "자산": list(opt.best_weights.keys()),
            "기본 비중(%)": [default_weights.get(k, 0) for k in opt.best_weights],
            "최적 비중(%)": [round(v * 100, 1) for v in opt.best_weights.values()],
        }
    )
    fig_bar = px.bar(
        weight_df.melt(id_vars="자산", var_name="구분", value_name="비중(%)"),
        x="자산",
        y="비중(%)",
        color="구분",
        barmode="group",
        title="기본 vs 최적 자산 비중",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(weight_df, use_container_width=True, hide_index=True)

    # ── Efficient Frontier scatter ───────────────────────────
    if opt.all_results is not None and not opt.all_results.empty:
        st.markdown("### 🗺️ Efficient Frontier")
        all_r = opt.all_results.copy()

        # Try to find volatility and return columns
        vol_col = next((c for c in all_r.columns if "vol" in c.lower()), None)
        ret_col = next((c for c in all_r.columns if "return" in c.lower()), None)
        sharpe_col = next((c for c in all_r.columns if "sharpe" in c.lower()), None)

        if vol_col and ret_col and sharpe_col:
            fig_ef = px.scatter(
                all_r,
                x=vol_col,
                y=ret_col,
                color=sharpe_col,
                color_continuous_scale="RdYlGn",
                labels={vol_col: "연간 변동성(%)", ret_col: "연간 수익률(%)", sharpe_col: "Sharpe"},
                title="Efficient Frontier (색상=Sharpe Ratio)",
                opacity=0.6,
            )
            # Mark the optimal point
            best_row = all_r.loc[all_r[sharpe_col].idxmax()]
            fig_ef.add_trace(
                go.Scatter(
                    x=[best_row[vol_col]],
                    y=[best_row[ret_col]],
                    mode="markers",
                    marker=dict(size=15, color="gold", symbol="star"),
                    name="최적점 ★",
                )
            )
            fig_ef.update_layout(height=500)
            st.plotly_chart(fig_ef, use_container_width=True)
        else:
            st.dataframe(all_r.head(20), use_container_width=True)

    # Export optimal weights
    st.markdown("### 📤 최적 비중 내보내기")
    csv_weights = weight_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 최적 비중 CSV 다운로드",
        data=csv_weights,
        file_name="optimal_weights.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────
# Education page
# ─────────────────────────────────────────────────────────────


def render_education_page():
    """Render educational content about the strategies."""
    st.header("🎓 전략 교육")

    with st.expander("📖 VAA 전략 개요", expanded=True):
        st.markdown(
            """
        ## VAA (Vigilant Asset Allocation)
        **Wouter Keller** (2017)이 개발한 전술적 자산배분 전략.

        ### 핵심 원칙
        1. **절대 모멘텀** — 모멘텀 점수 < 0 이면 위험 신호
        2. **상대 모멘텀** — 같은 자산군 내 최고 모멘텀 자산 선택
        3. **방어 전환** — 공격 자산 중 하나라도 음수 → 방어 자산으로 이동

        ### 모멘텀 공식
        ```
        Score = 12×(1개월 수익률) + 4×(3개월) + 2×(6개월) + 1×(12개월)
        ```
        단기 가중치를 높여 빠른 시장 반응을 확보합니다.
        """
        )

    with st.expander("🔮 OU 프로세스 예측"):
        st.markdown(
            r"""
        ## Ornstein-Uhlenbeck 프로세스

        평균 회귀 특성을 가진 확률 미분방정식:

        $$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$$

        | 파라미터 | 의미 |
        |---------|------|
        | θ (theta) | 평균 회귀 속도 |
        | μ (mu) | 장기 평균 |
        | σ (sigma) | 변동성 |

        Monte Carlo 시뮬레이션(1,000회)으로 미래 분포를 추정합니다.
        """
        )

    with st.expander("📊 기대 성과 (과거 백테스트 기준)"):
        st.markdown(
            """
        | 지표 | VAA | S&P 500 |
        |------|-----|---------|
        | CAGR | 12~15% | 10~12% |
        | Max Drawdown | 10~15% | 50%+ |
        | Sharpe Ratio | 1.0~1.5 | 0.5~0.7 |

        > ⚠️ **과거 성과가 미래를 보장하지 않습니다.**
        """
        )

    with st.expander("💡 퀀트 조언"):
        st.markdown(
            """
        1. **리밸런싱 주기** — 월 1회가 비용-효율 최적점
        2. **거래비용** — ETF 기준 0.05~0.1% 수준
        3. **과적합 주의** — Out-of-sample 검증 필수
        4. **모멘텀 붕괴** — 급격한 반전장(2009-03 등)에서 손실 가능
        """
        )


# ─────────────────────────────────────────────────────────────
# App entry point
# ─────────────────────────────────────────────────────────────


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title=UI.PAGE_TITLE,
        page_icon=UI.PAGE_ICON,
        layout=UI.LAYOUT,
        initial_sidebar_state="expanded",
    )

    init_session_state()
    page = render_sidebar()

    if page == "📊 VAA 분석":
        render_vaa_page()
    elif page == "💼 포트폴리오 관리":
        render_portfolio_page()
    elif page == "📈 전략 백테스트":
        render_backtest_page()
    elif page == "📉 리스크 분석":
        render_risk_page()
    elif page == "🔬 포트폴리오 최적화":
        render_optimizer_page()
    elif page == "🎓 전략 교육":
        render_education_page()

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Streamlit으로 제작 ❤️*")
    st.sidebar.markdown("*Yahoo Finance를 통한 실시간 데이터*")


if __name__ == "__main__":
    main()
