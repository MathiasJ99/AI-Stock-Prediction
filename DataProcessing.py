import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os


def GetHistorical():
    df = yf.download('^DJI', start='2013-01-01', end='2023-12-31')

    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')

    # Creates new column in DF called medium
    df['Medium'] = (df['High'] + df['Low']) / 2

    #df.to_csv("HistoricalDataCSV.csv")
    #df.to_excel('HistoricalData.xlsx') # uses $USD

    return df.head()


def GetEconomical():
    print(os.getenv("FRED_API_KEY"))


load_dotenv() #load env vars
GetEconomical()
#GetHistorical()
