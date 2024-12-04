import time
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from fredapi import Fred
from functools import reduce



def GetHistorical():
    Historicaldf = yf.download('^DJI', start='2013-01-01', end='2023-12-31')
    Historicaldf.reset_index(inplace=True)
    Historicaldf["Date"] = Historicaldf["Date"].dt.strftime('%Y-%m-%d')

    # Creates new column in DF called medium
    Historicaldf['Medium'] = (Historicaldf['High'] + Historicaldf['Low']) / 2

    #df.to_csv("HistoricalDataCSV.csv")
    #df.to_excel('HistoricalData.xlsx') # uses $USD

    return Historicaldf


def GetEconomical():
    #FRED API limits
    ## 120 Requests / minute
    ## 1000 records / request & maybe 100,000 observations / request

    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    start = "2020-01-01"
    end = "2021-12-31"

    All_data = []
    Economic_obj = ["GDP", "U2RATE"]

    #iterate though econ obj and get df of dates and data and append to All_data
    for obj in Economic_obj:
        result = fred.get_series(series_id=obj,observation_start=start, observation_end=end)
        resultdf = pd.DataFrame(result)
        resultdf = resultdf.reset_index()
        resultdf.columns = ["Dates", obj]
        All_data.append(resultdf)
        time.sleep(0.11)


    #merge data together on date
    EconomicDF = reduce(lambda left, right: pd.merge(left, right, on="Dates", how="inner"), All_data)
    print(EconomicDF.head)

    return EconomicDF




load_dotenv() #load env vars
GetEconomical()
#GetHistorical()
