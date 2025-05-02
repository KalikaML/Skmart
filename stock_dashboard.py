import pandas as pd
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# Function to fetch symbols from Excel file
def fetch_symbols(file_path):
    try:
        symbols = pd.read_csv(file_path)["Symbol"]
        return symbols.tolist()
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

# Function to fetch historical data for a symbol
def fetch_historical_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Main program
def main():
    st.title("Stock Data Fetcher")

    # Fetch symbols from Excel file
    symbol_file = "../EQUITY_L.csv"
    symbols = fetch_symbols(symbol_file)

    # Select symbol
    selected_symbol = st.selectbox("Select Symbol", symbols)

    # Set default date range
    end_date = date.today()
    default_start_date = end_date - timedelta(days=100)

    # Select date range
    start_date = st.date_input("Start Date", value=default_start_date)
    end_date = st.date_input("End Date", value=end_date)

    # Fetch and display data
    if st.button("Fetch Data"):
        df = fetch_historical_data(selected_symbol, start_date, end_date)
        if df is not None:
            st.write(df)
        else:
            st.write("No data available for the selected symbol and date range.")

if __name__ == "__main__":
    main()
