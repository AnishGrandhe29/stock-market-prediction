import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Page Config
st.set_page_config(page_title="NIFTY 50 Trading System", layout="wide")

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(base_dir, "backtest_results.csv")

# Load Data
@st.cache_data
def load_results():
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

# Import Live Monitor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitor.live_monitor import get_live_signal
import time

df = load_results()

if df is not None:
    st.title("NIFTY 50 Multimodal TCN Trading System")
    
    # Sidebar
    st.sidebar.header("Settings")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    
    # Filter
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]
    
    # Metrics
    total_return = filtered_df['Cumulative_Strategy'].iloc[-1] - 1
    market_return = filtered_df['Cumulative_Market'].iloc[-1] - 1
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy Return", f"{total_return:.2%}")
    col2.metric("Market Return", f"{market_return:.2%}")
    col3.metric("Outperformance", f"{total_return - market_return:.2%}")
    
    # Equity Curve
    st.subheader("Equity Curve")
    st.line_chart(filtered_df.set_index('Date')[['Cumulative_Strategy', 'Cumulative_Market']])
    
    # Drawdown
    st.subheader("Drawdown")
    cumulative = filtered_df['Cumulative_Strategy']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    st.line_chart(drawdown)
    
    # Recent Trades
    st.subheader("Recent Signals (Backtest)")
    st.dataframe(filtered_df[['Date', 'Close', 'Probability', 'Position']].tail(20))

    # --- Live Paper Trading Section ---
    st.markdown("---")
    st.header("Live Paper Trading & Online Learning")
    
    portfolio_path = os.path.join(base_dir, "portfolio_history.csv")
    trades_path = os.path.join(base_dir, "live_trades.csv")
    
    if os.path.exists(portfolio_path):
        st.subheader("Live Portfolio Performance")
        port_df = pd.read_csv(portfolio_path)
        port_df['Date'] = pd.to_datetime(port_df['Date'])
        
        st.line_chart(port_df.set_index('Date')['Value'])
        
        current_val = port_df['Value'].iloc[-1]
        st.metric("Current Portfolio Value", f"${current_val:,.2f}")
        
    if os.path.exists(trades_path):
        st.subheader("Live Trade Log")
        trades_df = pd.read_csv(trades_path)
        st.dataframe(trades_df)

    # --- Real-Time Monitor Section ---
    st.markdown("---")
    st.header("Real-Time NIFTY 50 Monitor")
    
    if st.button("Refresh Live Signal"):
        with st.spinner("Fetching live data and generating signal..."):
            try:
                signal_data = get_live_signal()
                
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Current Price", f"₹{signal_data['current_price']:,.2f}")
                m2.metric("Signal", signal_data['signal'], delta_color="normal" if "HOLD" in signal_data['signal'] else ("inverse" if "SELL" in signal_data['signal'] else "normal"))
                m3.metric("Probability (Up)", f"{signal_data['probability']:.2%}")
                
                st.info(f"Last Updated: {signal_data['last_updated']}")
                
                # Gauge Chart (Simple Progress Bar for now)
                st.write("Bullish Probability")
                st.progress(signal_data['probability'])
                
                if "BUY" in signal_data['signal']:
                    st.success("Recommendation: Strong Buying Opportunity detected based on TCN model.")
                elif "SELL" in signal_data['signal']:
                    st.error("Recommendation: Selling Pressure detected. Consider exiting positions.")
                else:
                    st.warning("Recommendation: Market is neutral or uncertain. Hold current positions.")
                    
            except Exception as e:
                st.error(f"Error fetching live signal: {e}")


    
else:
    st.error("Backtest results not found. Please run the backtest first.")
