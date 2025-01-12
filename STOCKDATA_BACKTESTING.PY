import pandas as pd
import os
import yfinance as yf
from datetime import date, timedelta
import numpy as np
import csv

#Function to fetch EOD data
def fetch_EOD():
    # Create the EOD60 folder if it doesn't exist
    output_folder = "EOD_BackTest"
    os.makedirs(output_folder, exist_ok=True)

    # Read the list of symbols from symbol.csv
    symbol_file = "EQUITY_L.csv"
    symbols = pd.read_csv(symbol_file)["Symbol"]

    # Set the end date as today
    end_date = date.today()

    # Set the start date as 60 days ago
    start_date = end_date - timedelta(days=700)

    # Fetch historical data for each symbol
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if not df.empty:
                # Save the data to a CSV file in the EOD folder
                filename = os.path.join(output_folder, f"{symbol}.csv")
                df.to_csv(filename)
                print(f"Data saved for {symbol} in {filename}")
            else:
                print(f"No data available for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    print("Data fetching and saving completed.")

 #-------- MAIN PROGRAM STARTS FROM HERE

# Directory containing CSV files
csv_directory = 'EOD_BackTest'
fetch_EOD()
