import streamlit as st
import pandas as pd
from datetime import date
from vaa_agg import get_performance, calculate_and_display_momentum
from port_ratio_calculator import calculate_capital_ratios
from rebalance import calculate_rebalance

def run_vaa_selection():
    """Run VAA aggregation and return the selected ETF."""
    try:
        aggressive_tickers = ['SPY', 'EFA', 'EEM', 'AGG']
        protective_tickers = ['LQD', 'IEF', 'SHY']
        calculation_date = date.today()

        # Process Aggressive Assets
        agg_performance = get_performance(aggressive_tickers, calculation_date)
        prot_performance = get_performance(protective_tickers, calculation_date)

        if agg_performance.empty:
            return None, None, None

        # Calculate momentum (simplified for UI)
        for df in [agg_performance, prot_performance]:
            if not df.empty:
                for col in ['1-Month', '3-Month', '6-Month', '12-Month']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(subset=['1-Month', '3-Month', '6-Month', '12-Month'], inplace=True)
                if not df.empty:
                    df['Momentum Score'] = (
                        df['1-Month'] * 12 + df['3-Month'] * 4 + 
                        df['6-Month'] * 2 + df['12-Month'] * 1
                    )

        agg_momentum = agg_performance.sort_values(by='Momentum Score', ascending=False) if not agg_performance.empty else pd.DataFrame()
        prot_momentum = prot_performance.sort_values(by='Momentum Score', ascending=False) if not prot_performance.empty else pd.DataFrame()

        # Selection logic
        selected_etf = None
        if not agg_momentum.empty:
            is_any_aggressive_negative = (agg_momentum['Momentum Score'] < 0).any()
            if is_any_aggressive_negative and not prot_momentum.empty:
                selected_etf = prot_momentum.index[0]
            elif not agg_momentum.empty:
                selected_etf = agg_momentum.index[0]

        return selected_etf, agg_momentum, prot_momentum
    except Exception as e:
        st.error(f"Error in VAA selection: {str(e)}")
        return None, None, None

def main():
    st.set_page_config(
        page_title="Portfolio Management System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Portfolio Management System")
    st.markdown("**VAA Strategy with Automated Rebalancing**")
    st.markdown("---")

    # Step 1: VAA Analysis (Must be first)
    st.subheader("📈 STEP 1: VAA ETF Selection")
    st.markdown("*First, we need to determine which ETF to allocate 50% of your portfolio to*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Run VAA Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing ETF momentum and performance..."):
                selected_etf, agg_momentum, prot_momentum = run_vaa_selection()
            
            if selected_etf:
                st.success(f"**🎯 VAA Selected ETF: {selected_etf}**")
                st.session_state.selected_etf = selected_etf
                
                # Store momentum data for display
                st.session_state.agg_momentum = agg_momentum
                st.session_state.prot_momentum = prot_momentum
                
                # Display selection logic
                if not agg_momentum.empty:
                    is_any_negative = (agg_momentum['Momentum Score'] < 0).any()
                    if is_any_negative:
                        st.info("🛡️ **Defensive Mode**: Negative momentum detected - selected protective asset")
                    else:
                        st.info("📈 **Growth Mode**: All aggressive assets positive - selected top performer")
                        
                st.markdown("---")
                st.markdown("✅ **Now you can proceed to enter your portfolio holdings**")
            else:
                st.error("❌ Could not select an ETF - insufficient data")
    
    with col2:
        st.markdown("**Strategy Overview:**")
        st.markdown("• 🎯 **50%** → VAA Selected ETF")
        st.markdown("• 📊 **12.5% each** → SPY, TLT, GLD, BIL")
        
    # Display momentum analysis if available
    if hasattr(st.session_state, 'agg_momentum') and not st.session_state.agg_momentum.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔥 Aggressive Assets:**")
            st.dataframe(st.session_state.agg_momentum.round(2), use_container_width=True)
        
        with col2:
            if hasattr(st.session_state, 'prot_momentum') and not st.session_state.prot_momentum.empty:
                st.markdown("**🛡️ Protective Assets:**")
                st.dataframe(st.session_state.prot_momentum.round(2), use_container_width=True)

    # Only show portfolio input after VAA is complete
    if hasattr(st.session_state, 'selected_etf'):
        st.markdown("---")
        st.subheader("💼 STEP 2: Enter Your Current Portfolio")
        st.markdown(f"*Enter your current holdings. Remember: **{st.session_state.selected_etf}** will get 50% allocation*")
        
        # Sidebar for portfolio input
        st.sidebar.header("📊 Current Portfolio Holdings")
        st.sidebar.markdown(f"**Selected ETF: {st.session_state.selected_etf}** (50% target)")
        st.sidebar.markdown("**Core Holdings:** SPY, TLT, GLD, BIL (12.5% each)")
        st.sidebar.markdown("---")
        
        col1, col2 = st.sidebar.columns(2)
        
        # Dynamic input based on selected ETF
        selected_etf = st.session_state.selected_etf
        
        with col1:
            # Selected ETF input (prominent)
            selected_shares = st.number_input(
                f"**{selected_etf} Shares** ⭐", 
                value=0, 
                min_value=0, 
                help=f"Your current {selected_etf} holdings (target: 50%)"
            )
            spy_shares = st.number_input("SPY Shares", value=0, min_value=0, help="S&P 500 ETF (target: 12.5%)")
            tlt_shares = st.number_input("TLT Shares", value=0, min_value=0, help="Long Treasury ETF (target: 12.5%)")
        
        with col2:
            gld_shares = st.number_input("GLD Shares", value=0, min_value=0, help="Gold ETF (target: 12.5%)")
            bil_shares = st.number_input("BIL Shares", value=0, min_value=0, help="Short Treasury ETF (target: 12.5%)")
        
        st.sidebar.markdown("---")
        additional_cash = st.sidebar.number_input(
            "Additional Cash ($)", 
            value=0.0, 
            min_value=0.0,
            help="Extra cash to add to your portfolio"
        )

        current_portfolio = {
            selected_etf: selected_shares,
            'SPY': spy_shares,
            'TLT': tlt_shares,
            'GLD': gld_shares,
            'BIL': bil_shares
        }

        # Portfolio Analysis
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Current Portfolio Analysis")
            
            try:
                # Only calculate if there are holdings
                has_any_holdings = any(shares > 0 for shares in current_portfolio.values())
                
                if has_any_holdings:
                    portfolio_breakdown = calculate_capital_ratios(current_portfolio)
                    if not portfolio_breakdown.empty:
                        # Format the dataframe for better display
                        display_df = portfolio_breakdown.copy()
                        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
                        display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:,.2f}")
                        display_df['Capital Ratio (%)'] = display_df['Capital Ratio (%)'].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        total_value = portfolio_breakdown['Market Value'].sum()
                        st.metric("💰 Total Portfolio Value", f"${total_value:,.2f}")
                        
                        # Show allocation vs target
                        if total_value > 0:
                            st.markdown("**Current vs Target Allocation:**")
                            for idx, row in portfolio_breakdown.iterrows():
                                if row['Shares'] > 0:
                                    current_pct = row['Capital Ratio (%)']
                                    target_pct = 50.0 if idx == selected_etf else 12.5
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.progress(current_pct/100, text=f"{idx}: {current_pct:.1f}%")
                                    with col_b:
                                        diff = current_pct - target_pct
                                        color = "🟢" if abs(diff) < 2 else "🟡" if abs(diff) < 5 else "🔴"
                                        st.write(f"Target: {target_pct}% {color}")
                    else:
                        st.warning("⚠️ Could not fetch price data for your holdings")
                else:
                    st.info("📊 Enter your portfolio holdings to see analysis")
            except Exception as e:
                st.error(f"❌ Error calculating portfolio: {str(e)}")

        with col2:
            st.subheader("🎯 Target Allocation Strategy")
            st.markdown(f"**Selected ETF: {selected_etf}**")
            st.progress(0.5, text=f"{selected_etf}: 50%")
            
            st.markdown("**Core Holdings:**")
            for etf in ['SPY', 'TLT', 'GLD', 'BIL']:
                st.progress(0.125, text=f"{etf}: 12.5%")
            
            if additional_cash > 0:
                st.info(f"💵 **${additional_cash:,.2f} additional cash** will be optimally allocated")

        # Rebalancing section
        st.markdown("---")
        st.subheader("⚖️ STEP 3: Portfolio Optimization & Rebalancing")
        
        # Check if portfolio has any holdings
        has_holdings = any(shares > 0 for shares in current_portfolio.values())
        
        if has_holdings or additional_cash > 0:
            if st.button("⚡ Calculate Optimal Rebalancing", type="primary", use_container_width=True):
                with st.spinner("🧮 Calculating optimal portfolio rebalancing..."):
                    try:
                        recommendations = calculate_rebalance(current_portfolio, selected_etf, additional_cash)
                        st.session_state.recommendations = recommendations
                    except Exception as e:
                        st.error(f"❌ Error calculating rebalancing: {str(e)}")
                        st.session_state.recommendations = None

            # Display recommendations if available
            if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
                recommendations = st.session_state.recommendations
                
                if 'error' not in recommendations:
                    # Summary metrics
                    st.markdown("### 📊 **Rebalancing Summary**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("💼 Current Value", f"${recommendations['current_value']:,.2f}")
                    with col2:
                        st.metric("💵 Additional Cash", f"${recommendations['additional_cash']:,.2f}")
                    with col3:
                        st.metric("🎯 Target Value", f"${recommendations['total_target_value']:,.2f}")
                    with col4:
                        final_value = recommendations.get('final_portfolio_value', 0)
                        st.metric("✅ Final Value", f"${final_value:,.2f}")
                    
                    # Transaction details
                    if recommendations['transactions']:
                        st.markdown("### 📋 **Required Transactions**")
                        
                        # Separate buy and sell transactions
                        sells = []
                        buys = []
                        
                        for ticker, trans in recommendations['transactions'].items():
                            if trans['action'] == 'SELL':
                                sells.append({
                                    'ETF': ticker,
                                    'Shares': f"{trans['shares']:,}",
                                    'Price': f"${trans['price']:.2f}",
                                    'Proceeds': f"${trans['proceeds']:,.2f}"
                                })
                            elif trans['action'] == 'BUY':
                                buys.append({
                                    'ETF': ticker,
                                    'Shares': f"{trans['shares']:,}",
                                    'Price': f"${trans['price']:.2f}",
                                    'Cost': f"${trans['cost']:,.2f}"
                                })
                        
                        col1, col2 = st.columns(2)
                        
                        if sells:
                            with col1:
                                st.markdown("**🔴 SELL Orders:**")
                                sells_df = pd.DataFrame(sells)
                                st.dataframe(sells_df, use_container_width=True, hide_index=True)
                        
                        if buys:
                            with col2:
                                st.markdown("**🟢 BUY Orders:**")
                                buys_df = pd.DataFrame(buys)
                                st.dataframe(buys_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ **Perfect Balance!** No transactions needed - your portfolio is optimally allocated!")
                    
                    # Final allocation analysis
                    if 'allocation_errors' in recommendations:
                        st.markdown("### 🎯 **Final Portfolio Allocation**")
                        
                        allocation_data = []
                        total_error = 0
                        
                        for ticker, error_data in recommendations['allocation_errors'].items():
                            shares = recommendations['optimized_portfolio'].get(ticker, 0)
                            pct_error = abs(error_data['percentage_error'])
                            total_error += pct_error
                            
                            # Status determination
                            if pct_error < 0.5:
                                status = "🟢 Perfect"
                            elif pct_error < 1.5:
                                status = "🟡 Good"
                            elif pct_error < 3.0:
                                status = "🟠 Fair"
                            else:
                                status = "🔴 Needs Work"
                            
                            allocation_data.append({
                                'ETF': ticker,
                                'Shares': f"{shares:,}",
                                'Value': f"${error_data['actual_value']:,.0f}",
                                'Target %': f"{error_data['target_percentage']:.1f}%",
                                'Actual %': f"{error_data['actual_percentage']:.1f}%",
                                'Error': f"{error_data['percentage_error']:+.1f}%",
                                'Status': status
                            })
                        
                        allocation_df = pd.DataFrame(allocation_data)
                        st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                        
                        # Optimization quality
                        avg_error = total_error / len(recommendations['allocation_errors'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📊 Avg Allocation Error", f"{avg_error:.2f}%")
                        
                        with col2:
                            if avg_error < 1.0:
                                quality = "🟢 Excellent"
                            elif avg_error < 2.5:
                                quality = "🟡 Good"
                            elif avg_error < 5.0:
                                quality = "🟠 Fair"
                            else:
                                quality = "🔴 Poor"
                            
                            st.markdown(f"**Optimization Quality:** {quality}")
                else:
                    st.error(f"❌ {recommendations['error']}")
        else:
            st.info("💡 Enter your current portfolio holdings or additional cash to proceed with optimization")

    else:
        # Show instructions when VAA hasn't been run yet
        st.markdown("---")
        st.info("🔍 **Please run VAA analysis first** to determine which ETF should receive 50% allocation")
        
        # Show strategy explanation
        with st.expander("📚 **Learn About the VAA Strategy**"):
            st.markdown("""
            **Vigilant Asset Allocation (VAA)** is a tactical asset allocation strategy that:
            
            1. **📊 Analyzes Momentum**: Uses 1, 3, 6, and 12-month performance data
            2. **🎯 Selects Assets**: Chooses between aggressive (SPY, EFA, EEM, AGG) and protective (LQD, IEF, SHY) assets
            3. **🛡️ Risk Management**: Switches to protective assets when negative momentum is detected
            4. **⚖️ Portfolio Balance**: Allocates 50% to selected asset, 12.5% each to core holdings
            
            **Possible Selected ETFs:**
            - **Aggressive**: SPY (S&P 500), EFA (Developed Markets), EEM (Emerging Markets), AGG (Bonds)
            - **Protective**: LQD (Corporate Bonds), IEF (Treasury Notes), SHY (Short Treasury)
            
            **Your Core Holdings** (always maintained at 12.5% each):
            - **SPY**: S&P 500 broad market exposure
            - **TLT**: Long-term Treasury bonds for stability
            - **GLD**: Gold for inflation protection
            - **BIL**: Short-term Treasury bills for liquidity
            """)

    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit • Real-time data via Yahoo Finance*")

if __name__ == "__main__":
    main()