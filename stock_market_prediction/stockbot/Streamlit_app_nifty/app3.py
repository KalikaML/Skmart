import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ta
import os
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure TensorFlow backend is initialized properly
tf.keras.backend.clear_session()

# Define the HybridModel class (unchanged)
class HybridModel:
    def __init__(self, bilstm_model, lgb_model, scaler):
        self.bilstm_model = bilstm_model
        self.lgb_model = lgb_model
        self.scaler = scaler

    def predict(self, latest_data):
        if len(latest_data) != 100:
            raise ValueError("Latest data must contain exactly 100 days")
        ohlc_cols = ["Open", "High", "Low", "Close"]
        latest_scaled = self.scaler.transform(latest_data[ohlc_cols])
        X_bilstm_live = latest_scaled.reshape(1, 100, 4)
        bilstm_pred = self.bilstm_model.predict(X_bilstm_live, verbose=0)
        latest_features = latest_data[["Prev_Close", "Volatility", "MA5", "MA20", "RSI", "Lagged_Return", "Volatility_Ratio"]].iloc[-1].values
        X_lgb_live = np.hstack([bilstm_pred, latest_features.reshape(1, -1)])
        prob = self.lgb_model.predict_proba(X_lgb_live)[0, 1]
        signal = "Buy" if prob > 0.7 else "Sell" if prob < 0.3 else "Hold"
        return prob, signal

# Load the hybrid model
model_path = 'hybrid_model_nifty_bank.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Please ensure it is in the same directory as this script.")
    st.stop()

try:
    hybrid_model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}. Ensure TensorFlow and Keras versions match the training environment.")
    st.stop()

def preprocess_data(df):
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required_cols].copy()
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    if len(df) < 120:
        raise ValueError("Data must contain at least 120 rows to compute indicators.")

    # Compute features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Volatility'] = df['High'] - df['Low']
    df['MA5'] = df['Close'].rolling(window=5).mean().shift(1)
    df['MA20'] = df['Close'].rolling(window=20).mean().shift(1)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().shift(1)
    df['Lagged_Return'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
    df['Volatility_Ratio'] = df['Volatility'] / df['Prev_Close']

    df = df.fillna(method='ffill').dropna()
    if len(df) < 100:
        raise ValueError("Not enough valid rows after preprocessing! Please provide more historical data.")
    
    return df.tail(100)

# Function for interactive trading-style graph
def plot_trading_analysis(df, prob, signal):
    # Create subplots: 2 rows (Candlestick + RSI)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=('OHLC with Moving Averages', 'RSI'),
                        row_heights=[0.7, 0.3])

    # Candlestick chart for OHLC
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='OHLC'),
                  row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], name='MA5', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20', line=dict(color='purple', width=1)), row=1, col=1)

    # RSI plot
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='blue', width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)")

    # Add prediction signal as an annotation on the last candlestick
    last_date = df['Date'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    signal_color = 'green' if signal == 'Buy' else 'red' if signal == 'Sell' else 'gray'
    fig.add_annotation(x=last_date, y=last_close, text=f"Signal: {signal} ({prob:.1%})", 
                       showarrow=True, arrowhead=1, ax=20, ay=-30, 
                       bgcolor=signal_color, font=dict(color='white'), row=1, col=1)

    # Update layout for interactivity and styling
    fig.update_layout(
        title='NIFTY Index Market Analysis',
        yaxis_title='Price',
        yaxis2_title='RSI',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,  # Disable default range slider for cleaner look
        template='plotly_dark',  # Dark theme for trading app feel
    )

    # Update axes for zooming and hovering
    fig.update_xaxes(
        rangeselector=dict(buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all")
        ])),
        row=2, col=1
    )

    return fig

# Streamlit UI
st.set_page_config(page_title="NIFTY Index 1% Profit Predictor", page_icon="📈", layout="wide")
st.title("NIFTY Index 1% Profit Predictor")
st.markdown("""
    Upload a CSV file with OHLC data for a NIFTY index to predict a 1% profit on the next trading day.
    Explore the interactive graph below to analyze market trends and the prediction.
""")

# Main content with file upload
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload NIFTY Index Data")
    st.markdown("Upload a CSV file with at least 120 days of OHLC data (Date,Open,High,Low,Close).")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="dynamic_uploader")

    # Process and predict whenever a new file is uploaded
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y', errors='coerce')
            df_processed = preprocess_data(df)
            prob, signal = hybrid_model.predict(df_processed)
            
            # Display prediction
            st.success(f"**Probability:** {prob:.4f}")
            st.success(f"**Signal:** {signal}")
            st.info(f"**Interpretation:** {signal} (Probability: {prob:.1%})")
            
            # Display interactive trading analysis
            st.subheader("Interactive Market Analysis")
            fig = plot_trading_analysis(df_processed, prob, signal)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with col2:
    st.subheader("Instructions")
    st.markdown("""
        - **Data Format**: At least 120 rows with columns: `Date`, `Open`, `High`, `Low`, `Close`.
        - **Date Format**: Use `DD Mon YYYY` (e.g., `28 Feb 2025`) in the CSV.
        - **Graph Features**:
          - Zoom in/out with mouse wheel or buttons (1m, 6m, All).
          - Hover over candlesticks to see OHLC, MA5, MA20, and RSI details.
          - Analyze trends with candlesticks, moving averages, and RSI.
        - **Prediction**: Signal and probability are shown on the graph and above.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model: Hybrid BiLSTM + LightGBM")