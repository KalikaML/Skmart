import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
from tensorflow.keras.models import load_model
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import streamlit as st

st.write("Current Working Directory:", os.getcwd())

# Function to load models efficiently with caching
@st.cache_resource()
def load_models():
    try:
        st.write("Current Working Directory:", os.getcwd())
        lstm = load_model('./lstm_model_all.h5', custom_objects={'mse': keras.losses.MeanSquaredError()})
        xgb = joblib.load('./xgb_model_all.pkl')
        scalers = joblib.load('./scalers.pkl')
        scaler_xgb = joblib.load('./scaler_xgb.pkl')
        return {"lstm": lstm, "xgb": xgb, "scalers": scalers, "scaler_xgb": scaler_xgb}
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load company and price data
@st.cache_data()
def load_data():
    try:
        company_df = pd.read_csv('./EQUITY_L.csv')
        price_df = pd.read_csv('./all_stocks_data.csv')
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df = price_df.groupby('Symbol').apply(add_features).reset_index(drop=True)
        price_df = price_df.dropna()
        
        # Rename column for consistency
        company_df.rename(columns={'Company Name': 'CompanyName'}, inplace=True)
        return company_df, price_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Feature engineering
def add_features(df):
    df['DailyReturn'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Target'] = (df['Close'].shift(-1) / df['Close'] >= 1.01).astype(int)
    return df

# Hybrid prediction function
def hybrid_prediction(symbol, price_df, models, days=7):
    if not models:
        return None, None
    
    stock_data = price_df[price_df['Symbol'] == symbol].sort_values('Date')
    if len(stock_data) < 20 or symbol not in models["scalers"]:
        return None, None
    
    lstm_model = models["lstm"]
    xgb_model = models["xgb"]
    scalers = models["scalers"]
    scaler_xgb = models["scaler_xgb"]
    
    # LSTM Prediction
    scaler = scalers[symbol]
    recent_data = stock_data['Close'].values[-20:]
    scaled_recent = scaler.transform(recent_data.reshape(-1, 1))
    X_pred = np.array([scaled_recent])
    lstm_pred = lstm_model.predict(X_pred)
    future_prices = scaler.inverse_transform(lstm_pred).flatten()[:days]

    # XGBoost Prediction
    features = ['Close', 'Volume', 'MA5', 'MA20', 'Volatility', 'DailyReturn']
    latest_features = stock_data[features].iloc[-1].values.reshape(1, -1)
    scaled_features = scaler_xgb.transform(latest_features)
    profit_prob = xgb_model.predict_proba(scaled_features)[0, 1]
    
    return future_prices, profit_prob

# Profit Calculator
def calculate_profit(initial_capital, predicted_prices):
    if not predicted_prices.any():
        return 0
    initial_price = predicted_prices[0]
    final_price = predicted_prices[-1]
    return round(((final_price - initial_price) / initial_price) * initial_capital, 2)

# Streamlit app interface
st.title("📈 Advanced Stock Market Prediction Dashboard")

company_df, price_df = load_data()
models = load_models()

if company_df is not None and 'CompanyName' in company_df.columns and 'Symbol' in company_df.columns:
    company_name = st.selectbox("Select a company", company_df['CompanyName'].unique())
    symbol = company_df.loc[company_df['CompanyName'] == company_name, 'Symbol'].values[0]
    st.write(f"Selected Company: {company_name}")
    st.write(f"Stock Symbol: {symbol}")

    # Custom prediction range
    days = st.slider("Select prediction range (days)", min_value=1, max_value=30, value=7)
    capital = st.number_input("Enter investment capital (₹)", min_value=1000, step=1000, value=10000)
    
    future_prices, profit_prob = hybrid_prediction(symbol, price_df, models, days=days)
    if future_prices is not None:
        st.success(f"✅ {days}-Day Price Prediction for {company_name} ({symbol})")
        
        future_dates = pd.date_range(start=pd.to_datetime(price_df['Date'].max()) + timedelta(days=1), periods=days)
        pred_table = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
        st.table(pred_table)
        
        projected_profit = calculate_profit(capital, future_prices)
        st.success(f"💰 Projected Profit: ₹{projected_profit} (if invested {capital} ₹ today)")
        st.success(f"📊 Probability of 1% Profit Today: {profit_prob:.2f}")
        
        # Additional Graphs
        st.subheader("📊 Stock Market Visualizations")
        fig1 = px.line(price_df[price_df['Symbol'] == symbol], x='Date', y='Close', title='Stock Price Over Time')
        fig2 = px.bar(price_df[price_df['Symbol'] == symbol], x='Date', y='Volume', title='Stock Volume Over Time')
        fig3 = px.line(price_df[price_df['Symbol'] == symbol], x='Date', y='DailyReturn', title='Daily Returns')
        fig4 = px.line(price_df[price_df['Symbol'] == symbol], x='Date', y='Volatility', title='Stock Volatility')
        
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)
    else:
        st.warning(f"⚠ No price data available for {company_name} ({symbol}). Prediction not possible.")
else:
    st.error("❌ Required columns 'CompanyName' or 'Symbol' not found in the dataset.")

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(symbol, price_df, models, days=7):
    stock_data = price_df[price_df['Symbol'] == symbol].sort_values('Date')
    # Get actual prices for the last 'days' days
    actual_prices = stock_data['Close'].values[-days:]

    # Get predicted prices
    predicted_prices, _ = hybrid_prediction(symbol, price_df, models, days=days)

    if predicted_prices is None:
        st.error("Not enough data to evaluate model accuracy.")
        return

    # Compute error metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

    st.write(f"📉 **Model Evaluation for {symbol}:**")
    st.write(f"🔹 **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"🔹 **Root Mean Squared Error (RMSE):** {rmse:.2f}")

    # Plot actual vs predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'].values[-days:], y=actual_prices, mode='lines+markers', name='Actual Price'))
    fig.add_trace(go.Scatter(x=stock_data['Date'].values[-days:], y=predicted_prices, mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
    
    st.plotly_chart(fig)

# Add this button in your Streamlit app
if st.button("Evaluate Model Accuracy"):
    evaluate_model(symbol, price_df, models, days=7)
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=price_df['Date'], y=price_df['MA5'], name='5-Day Moving Avg'))
fig_ma.add_trace(go.Scatter(x=price_df['Date'], y=price_df['MA20'], name='20-Day Moving Avg'))
fig_ma.update_layout(title="Stock Moving Averages", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_ma)
