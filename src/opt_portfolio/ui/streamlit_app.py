"""
Streamlit Web UI for Portfolio Management

This module provides an interactive web interface for the portfolio
management system with extended features and professional visualizations.

퀀트 관점:
- 시각화는 의사결정 품질 향상의 핵심
- 실시간 모니터링으로 빠른 대응 가능
- 인터랙티브 분석으로 전략 이해도 향상
"""

# Import from refactored modules
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from opt_portfolio.analysis.backtest import BacktestEngine  # noqa: E402
from opt_portfolio.config import ASSETS, UI  # noqa: E402
from opt_portfolio.core.cache import get_cache  # noqa: E402
from opt_portfolio.core.portfolio import Portfolio  # noqa: E402
from opt_portfolio.strategies.momentum import MomentumAnalyzer  # noqa: E402
from opt_portfolio.strategies.vaa import VAAStrategy  # noqa: E402


def init_session_state():
    """Initialize Streamlit session state."""
    if "selected_etf" not in st.session_state:
        st.session_state.selected_etf = None
    if "vaa_result" not in st.session_state:
        st.session_state.vaa_result = None
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = None


def render_sidebar():
    """Render sidebar with navigation and info."""
    st.sidebar.title("🚀 포트폴리오 관리")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "페이지 이동",
        ["📊 VAA 분석", "💼 포트폴리오 관리", "📈 전략 백테스트", "🎓 전략 교육"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 요약 정보")
    st.sidebar.markdown("""
    **VAA 전략:**
    - 50% → 선택된 ETF
    - 12.5%씩 → SPY, TLT, GLD, BIL
    
    **리밸런싱:** 월 1회
    """)

    # Cache status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 캐시 상태")
    cache = get_cache()
    stats = cache.get_cache_stats()
    if not stats.empty:
        st.sidebar.metric("캐시된 티커 수", len(stats))
    else:
        st.sidebar.info("캐시 데이터 없음")

    return page


def render_vaa_page():
    """Render VAA analysis page with extended features."""
    st.header("📊 VAA ETF 선택 분석")
    st.markdown("*고급 예측 기반 Vigilant Asset Allocation*")

    # 분석 설정
    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_date = st.date_input("분석 기준일", value=date.today(), max_value=date.today())

    with col2:
        strategy = st.selectbox(
            "선택 전략",
            ["현재 모멘텀(VAA)", "1개월 예측", "3개월 예측", "모멘텀 변화(Δ)"],
            help="최적의 자산 선택 기준을 고르세요",
        )

    with col3:
        show_forecast = st.checkbox("승률(Win Prob.) 보기", value=True)

    # 분석 실행
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

            except Exception as e:
                st.error(f"분석 실패: {e}")
                return

    # 결과 표시
    if st.session_state.vaa_result:
        result = st.session_state.vaa_result

        # 모드 표시
        if result.is_defensive:
            st.warning("🛡️ **방어 모드** - 공격 자산군에 음의 모멘텀 감지")
        else:
            st.success("📈 **성장 모드** - 모든 공격 자산군이 양의 모멘텀")

        # 선택된 ETF
        st.markdown(f"### 🎯 선택된 ETF: **{result.selected_etf}**")

        # 랭킹 표시
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

        # 전략 추천
        if result.strategy_recommendations:
            st.markdown("### 📊 전략별 추천")

            rec_data = []
            for strategy, data in result.strategy_recommendations.items():
                if "asset" in data:
                    rec_data.append(
                        {
                            "전략": strategy,
                            "추천 자산": data["asset"],
                            "점수": round(data["score"], 2),
                        }
                    )

            if rec_data:
                st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

        # 승률 차트
        if hasattr(st.session_state, "win_probs") and not st.session_state.win_probs.empty:
            st.markdown("### 🎲 승률(다음달 기준)")

            fig = px.bar(
                x=st.session_state.win_probs.values * 100,
                y=st.session_state.win_probs.index,
                orientation="h",
                labels={"x": "승률(%)", "y": "자산"},
                title="최고 성과 자산 확률",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # 모멘텀 히스토리 차트
        render_momentum_chart()


def render_momentum_chart():
    """Render historical momentum chart."""
    st.markdown("### 📈 모멘텀 히스토리")

    col1, col2 = st.columns([3, 1])

    with col1:
        years = st.slider("조회 기간(년)", 1, 5, 2)

    with col2:
        asset_type = st.radio("자산군 선택", ["공격", "방어"])

    tickers = (
        list(ASSETS.AGGRESSIVE_TICKERS) if asset_type == "공격" else list(ASSETS.PROTECTIVE_TICKERS)
    )

    momentum_analyzer = MomentumAnalyzer()
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)

    momentum_df = momentum_analyzer.calculate_historical_momentum(tickers, start_date, end_date)

    if not momentum_df.empty:
        fig = go.Figure()

        for ticker in momentum_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=momentum_df.index,
                    y=momentum_df[ticker],
                    mode="lines",
                    name=ticker,
                    hovertemplate="%{y:.1f}",
                )
            )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0선")

        fig.update_layout(
            title=f"{asset_type} 자산군 모멘텀 히스토리",
            xaxis_title="날짜",
            yaxis_title="모멘텀 점수",
            legend_title="자산",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)


def render_portfolio_page():
    """Render portfolio management page."""
    st.header("💼 포트폴리오 관리")

    # VAA 분석 여부 확인
    if not st.session_state.selected_etf:
        st.warning("⚠️ 먼저 VAA 분석을 실행해 ETF를 선택하세요.")
        return

    selected_etf = st.session_state.selected_etf

    st.success(f"🎯 VAA 선택 ETF: **{selected_etf}** (목표 비중 50%)")

    # 포트폴리오 입력
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

    # 포트폴리오 생성
    portfolio = Portfolio.from_dict(holdings)
    portfolio.update_prices()

    # 현재 배분 표시
    if any(s > 0 for s in holdings.values()):
        st.markdown("### 📊 현재 배분 현황")

        allocation = portfolio.get_allocation()
        if not allocation.empty:
            # 파이 차트
            fig = px.pie(
                allocation.reset_index(),
                values="Value",
                names="Ticker",
                title="현재 포트폴리오 배분",
            )
            st.plotly_chart(fig, use_container_width=True)

            # 배분 테이블
            st.dataframe(allocation.round(2), use_container_width=True)

            st.metric("💰 총 포트폴리오 가치", f"${portfolio.total_value:,.2f}")

    # 리밸런싱
    st.markdown("### ⚖️ 포트폴리오 리밸런싱")

    if st.button("⚡ 최적 리밸런싱 계산", type="primary", use_container_width=True):
        with st.spinner("최적 거래 계산 중..."):
            recommendations = portfolio.calculate_rebalance(selected_etf, additional_cash)

            if "error" in recommendations:
                st.error(recommendations["error"])
            else:
                render_rebalancing_results(recommendations)


def render_rebalancing_results(recommendations: dict):
    """Render rebalancing recommendations."""
    # 요약 지표
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("현재 가치", f"${recommendations['current_value']:,.2f}")
    with col2:
        st.metric("추가 투자금", f"${recommendations['additional_cash']:,.2f}")
    with col3:
        st.metric("목표 가치", f"${recommendations['total_target_value']:,.2f}")
    with col4:
        st.metric("최종 가치", f"${recommendations['final_portfolio_value']:,.2f}")

    # 거래 내역
    if recommendations["transactions"]:
        st.markdown("### 📋 필요 거래 내역")

        col1, col2 = st.columns(2)

        sells = []
        buys = []

        for ticker, trans in recommendations["transactions"].items():
            if trans["action"] == "SELL":
                sells.append(
                    {
                        "ETF": ticker,
                        "주식 수": trans["shares"],
                        "가격": f"${trans['price']:.2f}",
                        "매도금액": f"${trans['proceeds']:,.2f}",
                    }
                )
            else:
                buys.append(
                    {
                        "ETF": ticker,
                        "주식 수": trans["shares"],
                        "가격": f"${trans['price']:.2f}",
                        "매수금액": f"${trans['cost']:,.2f}",
                    }
                )

        with col1:
            if sells:
                st.markdown("**🔴 매도 주문**")
                st.dataframe(pd.DataFrame(sells), use_container_width=True, hide_index=True)

        with col2:
            if buys:
                st.markdown("**🟢 매수 주문**")
                st.dataframe(pd.DataFrame(buys), use_container_width=True, hide_index=True)
    else:
        st.success("✅ 포트폴리오가 최적화되어 거래가 필요 없습니다.")

    # 배분 오차
    st.markdown("### 🎯 최종 배분 분석")

    error_data = []
    for ticker, error in recommendations["allocation_errors"].items():
        pct_error = abs(error["percentage_error"])
        status = "🟢" if pct_error < 1 else "🟡" if pct_error < 3 else "🔴"

        error_data.append(
            {
                "ETF": ticker,
                "주식 수": recommendations["optimized_portfolio"].get(ticker, 0),
                "목표 %": f"{error['target_percentage']:.1f}%",
                "실제 %": f"{error['actual_percentage']:.1f}%",
                "오차": f"{error['percentage_error']:+.1f}%",
                "상태": status,
            }
        )

    st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)

    # 최적화 품질
    total_error = sum(
        abs(e["percentage_error"]) for e in recommendations["allocation_errors"].values()
    )
    avg_error = total_error / len(recommendations["allocation_errors"])

    if avg_error < 1:
        st.success(f"🟢 배분 품질 우수 (평균 오차: {avg_error:.2f}%)")
    elif avg_error < 2.5:
        st.info(f"🟡 배분 품질 양호 (평균 오차: {avg_error:.2f}%)")
    else:
        st.warning(f"🟠 더 나은 배분을 위해 추가 자본 투입 권장 (평균 오차: {avg_error:.2f}%)")


def render_backtest_page():
    """Render backtesting page."""
    st.header("📈 전략 백테스트")
    st.markdown("*VAA 전략을 과거 데이터로 비교 분석*")

    # 백테스트 설정
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
            engine = BacktestEngine()
            results = engine.run_vaa_backtest(years=years, strategies=strategies)

            if results:
                st.session_state.backtest_results = results

    # 결과 표시
    if hasattr(st.session_state, "backtest_results"):
        results = st.session_state.backtest_results

        # 요약 테이블
        st.markdown("### 📊 성과 요약")

        summary_data = []
        for name, result in results.items():
            summary_data.append(
                {
                    "전략": name,
                    "최종 가치": f"${result.final_capital:,.0f}",
                    "CAGR": f"{result.cagr:.1%}",
                    "샤프지수": f"{result.sharpe_ratio:.2f}",
                    "최대 낙폭": f"{result.max_drawdown:.1%}",
                    "승률": f"{result.win_rate:.1%}",
                }
            )

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # 자본곡선
        st.markdown("### 📈 자본곡선 비교")

        fig = go.Figure()
        for name, result in results.items():
            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    mode="lines",
                    name=name,
                )
            )

        fig.update_layout(
            title="전략별 자본곡선 비교",
            xaxis_title="날짜",
            yaxis_title="포트폴리오 가치($)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # 낙폭 차트
        st.markdown("### 📉 낙폭 분석")

        fig = go.Figure()
        for name, result in results.items():
            running_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - running_max) / running_max * 100
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index, y=drawdown.values, mode="lines", name=name, fill="tozeroy"
                )
            )

        fig.update_layout(
            title="전략별 낙폭 비교", xaxis_title="날짜", yaxis_title="낙폭(%)", height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def render_education_page():
    """Render educational content about the strategies."""
    st.header("🎓 전략 교육")

    st.markdown("""
    ## VAA (Vigilant Asset Allocation) 전략
    
    ### 📖 개요
    VAA는 **Wouter Keller**가 2017년에 개발한 전술적 자산배분 전략입니다.
    
    ### 🎯 핵심 원칙
    
    1. **절대 모멘텀 (Absolute Momentum)**
       - 자산의 모멘텀 점수가 0 이상인지 확인
       - 음수면 "위험 신호"로 간주
    
    2. **상대 모멘텀 (Relative Momentum)**
       - 같은 자산군 내에서 가장 높은 모멘텀 자산 선택
    
    3. **방어 전환 (Defensive Switch)**
       - 공격 자산 중 하나라도 음의 모멘텀 → 방어 자산으로 전환
    
    ### 📊 모멘텀 계산
    
    ```
    Momentum Score = 12×(1개월 수익률) + 4×(3개월 수익률) + 2×(6개월 수익률) + 1×(12개월 수익률)
    ```
    
    단기 수익률에 높은 가중치를 부여하여 빠른 시장 반응 확보
    
    ### 🔮 OU 프로세스 예측
    
    **Ornstein-Uhlenbeck 프로세스**는 평균 회귀 특성을 모델링:
    
    $$dX_t = \\theta(\\mu - X_t)dt + \\sigma dW_t$$
    
    - θ (theta): 평균 회귀 속도
    - μ (mu): 장기 평균
    - σ (sigma): 변동성
    
    ### 💡 퀀트 조언
    
    1. **리밸런싱 주기**: 월 1회가 비용-효율 최적
    2. **거래비용**: ETF는 0.1% 수준으로 낮음
    3. **슬리피지**: 유동성 높은 ETF 사용으로 최소화
    4. **과적합 주의**: Out-of-sample 테스트 필수
    
    ### 📈 기대 성과 (15년 백테스트 기준)
    
    | 지표 | VAA | S&P 500 |
    |------|-----|---------|
    | CAGR | 12-15% | 10-12% |
    | Max Drawdown | 10-15% | 50%+ |
    | Sharpe Ratio | 1.0-1.5 | 0.5-0.7 |
    
    ### ⚠️ 주의사항
    
    - **과거 성과가 미래를 보장하지 않음**
    - 모멘텀 붕괴(Momentum Crash) 위험 존재
    - 시장 구조 변화 시 전략 효과 감소 가능
    """)


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
    elif page == "🎓 전략 교육":
        render_education_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Streamlit으로 제작 ❤️*")
    st.sidebar.markdown("*Yahoo Finance를 통한 실시간 데이터*")


if __name__ == "__main__":
    main()
