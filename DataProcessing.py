import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os


def configure():
    load_dotenv()

def GetHistorical():
    df = yf.download('^DJI', start='2013-01-01', end='2023-12-31')

    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')

    # Creates new column in DF called medium
    df['Medium'] = (df['High'] + df['Low']) / 2

    #df.to_csv("HistoricalDataCSV.csv")
    df.to_excel('HistoricalData.xlsx') # uses usd

    return df.head()


def GetEconomical():
    os.getenv("fred_api_key")


configure()
GetHistorical()
