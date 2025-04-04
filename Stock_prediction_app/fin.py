import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import plotly.graph_objs as go

# Cache data loading for performance
@st.cache_data
def load_data():
    equity_data = pd.read_csv('EQUITY_L.csv')
    price_data = pd.read_csv('all_stocks_data.csv')
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    return equity_data, price_data

# Cache model loading for performance
@st.cache_resource
def load_models():
    lstm_model = load_model('lstm_model_all.h5')
    xgb_model = joblib.load('xgb_model_all.pkl')
    scalers = joblib.load('scalers.pkl')
    scaler_xgb = joblib.load('scaler_xgb.pkl')
    return lstm_model, xgb_model, scalers, scaler_xgb

# Add technical indicators to price data
def add_features(df):
    df['DailyReturn'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    return df.dropna()

# Hybrid prediction function (simplified example)
def hybrid_prediction(stock_data, lstm_model, xgb_model, scalers, scaler_xgb, days=7):
    # Prepare data for LSTM (last 20 days)
    scaler = scalers[stock_data['Symbol'].iloc[0]]
    last_20 = stock_data['Close'].tail(20).values.reshape(-1, 1)
    scaled_data = scaler.transform(last_20)
    X = np.array([scaled_data])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # LSTM prediction
    lstm_pred = lstm_model.predict(X)
    lstm_pred = scaler.inverse_transform(lstm_pred)[0]
    
    # Prepare data for XGBoost
    features = stock_data.tail(1)[['Close', 'Volume', 'MA5', 'MA20', 'Volatility', 'DailyReturn']]
    scaled_features = scaler_xgb.transform(features)
    
    # XGBoost prediction (profit probability)
    profit_prob = xgb_model.predict_proba(scaled_features)[:, 1][0]
    
    return lstm_pred[:days], profit_prob

# Main app
st.title("Advanced Stock Market Prediction Dashboard")
equity_data, price_data = load_data()
lstm_model, xgb_model, scalers, scaler_xgb = load_models()

# Display last updated date
st.write(f"Data last updated on: {price_data['Date'].max().strftime('%Y-%m-%d')}")

# Company selection
company = st.selectbox("Select Company", equity_data['CompanyName'])
symbol = equity_data[equity_data['CompanyName'] == company]['Symbol'].iloc[0]
stock_data = price_data[price_data['Symbol'] == symbol].copy()
stock_data = add_features(stock_data)

# Prediction settings
days = st.slider("Prediction Days", 1, 7, 7)
capital = st.number_input("Investment Capital (INR)", min_value=1000, value=10000)

if st.button("Predict"):
    pred_prices, profit_prob = hybrid_prediction(stock_data, lstm_model, xgb_model, scalers, scaler_xgb, days)
    future_dates = pd.date_range(start=stock_data['Date'].max() + pd.Timedelta(days=1), periods=days)
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': pred_prices})
    
    st.subheader("Predictions")
    st.table(pred_df)
    st.write(f"Profit Probability: {profit_prob:.2%}")
    
    # Plot historical and predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, name="Predicted", line=dict(dash='dash')))
    fig.update_layout(title=f"{company} Stock Price", xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig)