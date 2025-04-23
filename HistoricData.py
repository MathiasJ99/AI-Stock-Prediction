import time
import numpy as np
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os
from functools import reduce



load_dotenv() #load env vars
start = os.getenv("START")
end = os.getenv("END")


def GetHistoric(file_name, ticker):
    historic_path = "HistoricalData_"+ticker+".xlsx"
    historic_technical_path = "HistoricalAndTechnicalData_"+ticker+".xlsx"

    if file_name == historic_path: ## assuming data with only OHLCV and dates
        HistoricalDF = pd.read_excel(file_name)
        HistoricalDF['Medium'] = (HistoricalDF['High'] + HistoricalDF['Low']) / 2

        HistoricalDF_and_TechnicalDF = GetTechnicalIndicators(HistoricalDF)

        print(HistoricalDF_and_TechnicalDF.head())
        HistoricalDF_and_TechnicalDF.to_excel(historic_technical_path)  # uses $USD
        return HistoricalDF_and_TechnicalDF

    elif file_name == historic_technical_path:
        HistoricalDF_and_TechnicalDF = pd.read_excel(file_name)
        return HistoricalDF_and_TechnicalDF

    else:
        HistoricalDF = yf.download(ticker, start=start, end=end)

        # Creates new column in df called medium
        HistoricalDF['Medium'] = (HistoricalDF['High'] + HistoricalDF['Low']) / 2

        # formatting
        ##converting multi-dimensional header to 1d header
        one_dim_headers = [col[0] for col in HistoricalDF.columns.values]
        HistoricalDF.columns = one_dim_headers  # update the DataFrame headers
        HistoricalDF = HistoricalDF.reset_index()
        HistoricalDF.to_excel(historic_path)

        HistoricalDF_and_TechnicalDF = GetTechnicalIndicators(HistoricalDF)
        print(HistoricalDF_and_TechnicalDF.head())
        HistoricalDF_and_TechnicalDF.to_excel(historic_technical_path)  # uses $USD
        return HistoricalDF_and_TechnicalDF

def GetTechnicalIndicators(df): ## need to calculate MACD and VIX
    ###Trend indicators
    #EMA-12
    df["12_day_EMA"] = df['Close'].ewm(span=12, adjust=False).mean()

    #EMA-26
    df["26_day_EMA"] = df['Close'].ewm(span=26, adjust=False).mean()

    ##ADX average directional index
    # Calculate True Range (TR)
    df['previous_close'] = df['Close'].shift(1)
    df['high-low'] = df['High'] - df['Low']
    df['high-prev_close'] = abs(df['High'] - df['previous_close'])
    df['low-prev_close'] = abs(df['Low'] - df['previous_close'])
    df['TR'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
    period = 14  ## using rolling window calculation so first period values will be Nan

    # Calculate +DM and -DM
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),df['Low'].shift(1) - df['Low'], 0)

    # Smooth the TR, +DM, and -DM using the initial SMA
    df['TR_smooth'] = df['TR'].rolling(window=period, min_periods=1).mean()
    df['+DM_smooth'] = df['+DM'].rolling(window=period, min_periods=1).mean()
    df['-DM_smooth'] = df['-DM'].rolling(window=period, min_periods=1).mean()

    # Calculate +DI and -DI
    df['+DI'] = (df['+DM_smooth'] / df['TR_smooth']) * 100
    df['-DI'] = (df['-DM_smooth'] / df['TR_smooth']) * 100

    # Compute DX
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (abs(df['+DI'] + df['-DI']))) * 100

    # Initialize ADX with SMA for the first 14 periods
    df['ADX'] = np.nan
    df.loc[period - 1, 'ADX'] = df['DX'][:period].mean()

    # Apply Wilder's Smoothing for ADX
    for i in range(period, len(df)):
        df.loc[i, 'ADX'] = (df.loc[i - 1, 'ADX'] * (period - 1) + df.loc[i, 'DX']) / period

    df["ADX"] = df["ADX"].bfill()
    # Drop unnecessary columns
    df.drop(['previous_close', 'high-low', 'high-prev_close', 'low-prev_close',  '+DI', '-DI', '+DM', '-DM','TR_smooth', '+DM_smooth', '-DM_smooth', 'DX'], axis=1, inplace=True)



    ###Volatility Indicators
    # ATR Average true range
    df["SMA_ATR"] = df['TR'].rolling(window=14, min_periods=1).mean()
    df["EMA_ATR"] = df['TR'].ewm(span=14, adjust=False).mean()

    df.drop(['TR'], axis=1, inplace=True)

    #VIX measured by FRED so included in EconomicData.py


    ###Momentum indicators
    ##RSI
    period = 14  # Standard RSI period
    # Calculate daily price changes
    df['delta'] = df['Close'].diff()

    # Separate gains and losses
    df['gain'] = df['delta'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['delta'].apply(lambda x: abs(x) if x < 0 else 0)

    # Calculate average gains and losses
    df['avg_gain'] = df['gain'].rolling(window=period, min_periods=1).mean()
    df['avg_loss'] = df['loss'].rolling(window=period, min_periods=1).mean()

    #Calculate RS & RSI
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1 + df['rs']))

    #Drop intermediate columns
    df.drop(['delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1, inplace=True)

    #drop first 13 values
    df.loc[:13,"rsi"] =np.nan
    df["rsi"] = df['rsi'].fillna(df['rsi'].mean())

    #MACD
    df['MACD'] = df['12_day_EMA'] - df['26_day_EMA']



    ###Volume based indicators
    #OBV
    df["OBV"] = (np.where(df["Close"] > df["Close"].shift(1), df["Volume"],np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0))).cumsum()
    #VWAP
    df["Cum_Vol"] = df["Volume"].cumsum()
    df["Cum_PV"] = (df["Close"] * df["Volume"]).cumsum()
    df["VWAP"] = df["Cum_PV"] / df["Cum_Vol"]

    df.drop(['Cum_Vol', 'Cum_PV'], axis=1, inplace=True)

    df.ffill()

    return df




