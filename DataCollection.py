import time
import numpy as np
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os
from fredapi import Fred
from functools import reduce
from SentimentalData import *
from HistoricData import *
from EconomicData import *


load_dotenv() #load env vars
start = os.getenv("START")
end = os.getenv("END")

def Merge(HistoricalDF, EconomicalDF, SentimentalDF):
    # make Date column datetime objects
    HistoricalDF['Date'] = pd.to_datetime(HistoricalDF['Date'])
    EconomicalDF['Date'] = pd.to_datetime(EconomicalDF['Date'])
    SentimentalDF['Date'] = pd.to_datetime(SentimentalDF['Date'])

    HistoricalAndEconomicalDF = pd.merge(HistoricalDF, EconomicalDF, on="Date", how="inner")
    #HistoricalAndEconomicalDF.to_excel("HistoricalAndEconomicalDF.xlsx")

    MergedDF = pd.merge(HistoricalAndEconomicalDF, SentimentalDF, on="Date",  how="left")

    ##Formatting
    MergedDF = MergedDF.drop(columns = ["Unnamed: 0_y", "Unnamed: 0_x"])
    MergedDF['Date'] = MergedDF['Date'].dt.date

    MergedDF.to_excel('MergedDF.xlsx')  # uses $USD
    return MergedDF

def Clean(df):
    #dropping first 14 values because missing data from calculating ADX,SMA_ATR, EMA_ATR, rsi, OBV (only first value)
    df = df.iloc[14:].reset_index(drop=True)

    #dropping any other row if it has at least one missing value
    df = df.dropna().reset_index(drop=True)
    df.to_excel('MergedAndCleanedDF.xlsx')  # uses $USD
    return df

def GetData():
    HistoricDF = pd.read_excel("HistoricalAndTechnicalData.xlsx") # GetHistoric("Google_Stock_Train (2010-2022).csv") or GetHistoric() if using yfinance
    EconomicDF = pd.read_excel("EconomicData.xlsx") #GetEconomic()
    SentimentalDF =  pd.read_excel("Sentimental_google_sorted_nltk_scores.xlsx") #GetSentimental()

    MergedDF = Merge(HistoricDF, EconomicDF,  SentimentalDF)
    #CleanedDF = Clean(MergedDF)

    return MergedDF

#testing
print(GetData().head())

