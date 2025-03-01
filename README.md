# Skmart
### Project
problem statement : Build Trading bot which will buy and sell equity stocks for 1% profit daily on capital invesment along with it
                     it will give the user complete overview and news analysis of equity on daily basis before market opens.


Steps : 
   1) Train the ML model on nifty data.
   2) Which equity stocks have probability of giving 1% profit.
   3) Kite API access and understand buy and sell process.

      

### Approaches 
1. For all symbols or stocks in index one generalize ML model: 10 symbols with nifty weightage,
  1)  Identify  10 high volume of stocks for the next week and train: explore [screener](https://github.com/pranjal-joshi/Screeni-py)

 


### Prediction  

###  1. Technical Indicators

Use technical indicators to gauge momentum:

Relative Strength Index (RSI): Measures the speed and change of price movements. An RSI above 70 indicates overbought conditions, while below 30 suggests oversold conditions2.

Moving Average Convergence Divergence (MACD): Signals momentum changes when the MACD line crosses the signal line2.

Average Directional Index (ADX): Measures trend strength; values above 50 indicate strong momentum2.

### 2. Momentum Indicators
Price Change: Look for stocks with significant recent price increases.

Trading Volume: High trading volumes often accompany momentum stocks7.

### 3. Market Trends and Sectors
Identify sectors with strong momentum:

Sectoral Analysis: Focus on trending sectors like tech or renewable energy16.

Index Performance: Check if sector-specific indices are showing upward momentum6.

### 4. Stock Screeners
Utilize stock screeners to filter stocks based on momentum criteria:

Screener.in: Offers pre-built screens for momentum stocks58.

Tickertape: Allows you to create custom screens based on momentum indicators4.

### 5. News and Events
Monitor news and events that could impact stock prices:

Earnings Reports: Stocks often experience price surges after positive earnings surprises1.

Product Launches: New product announcements can drive short-term momentum1


week1 :
Resource collection and environment set

Automate and partial

10 yrs historical data
Tech nifty50
1) Daily Timeframe customized EMA ,RACD & RSI
2) Daily to Week conversion and same indicator

positive cross over

Analysis
3) Trends for high return
4) Bollinger Band Std deviation + candlestick


Deep week2
Deployment and backtesting


1) Dashboard for next day
    1) ML + Indicator

Task
1) historical data fetch ML train model

ReAL TIME FEEDING AND PREDICTION

2) Real time testing



2) 1% Bot

3)
