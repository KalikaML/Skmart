import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Load existing data
df = pd.read_csv('all_stocks_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Get unique stock symbols
symbols = df['Symbol'].unique()

# List to store new data
new_data_list = []

# Fetch new data for each symbol
for symbol in symbols:
    last_date = df[df['Symbol'] == symbol]['Date'].max()
    start_date = last_date + timedelta(days=1)
    end_date = datetime.today()
    
    if start_date < end_date:
        try:
            new_data = yf.download(symbol + '.NS', start=start_date, end=end_date)
            if not new_data.empty:
                new_data = new_data.reset_index()
                new_data['Symbol'] = symbol
                new_data = new_data[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                new_data_list.append(new_data)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

# Append and save new data if available
if new_data_list:
    new_data_df = pd.concat(new_data_list, ignore_index=True)
    updated_df = pd.concat([df, new_data_df], ignore_index=True)
    updated_df = updated_df.sort_values(by=['Symbol', 'Date'])
    updated_df.to_csv('all_stocks_data.csv', index=False)
    print("Stock data successfully updated!")
else:
    print("No new data available.")