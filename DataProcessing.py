import time
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os
from fredapi import Fred
from functools import reduce
import openpyxl


start = "2020-01-01"
end = "2020-06-30"

def GetHistorical():
    HistoricalDF = yf.download('^DJI', start=start, end=end)
    # Creates new column in DF called medium
    HistoricalDF['Medium'] = (HistoricalDF['High'] + HistoricalDF['Low']) / 2

    #formatting
    ##converting multi dimensonal header to 1d header
    one_dim_headers = [col[0] for col in HistoricalDF.columns.values]
    HistoricalDF.columns = one_dim_headers# Update the DataFrame headers
    HistoricalDF = HistoricalDF.reset_index()
    HistoricalDF.rename(columns={"Date": "Dates"}, inplace=True)

    #testing
    #HistoricalDF.to_excel('HistoricalData.xlsx') # uses $USD
    print(HistoricalDF.head())
    return HistoricalDF


def GetEconomical():
    Daily_Economic_obj = ["DFF","DFII5","DFII10","T5YIE","T10YIE"]
    Monthly_Economic_obj = ["UNRATE","CPIAUCSL","A229RX0","INDPRO"]
    Quarterly_Economic_obj = ["GDP", "A939RX0Q048SBEA"]

    Daily_data_dfs = APICall(Daily_Economic_obj)
    Monthly_data_dfs = APICall(Monthly_Economic_obj)
    Quarterly_Data_dfs = APICall(Quarterly_Economic_obj)

    #merge data together on date
    DailyDF = reduce(lambda left, right: pd.merge(left, right, on="Dates", how="inner"), Daily_data_dfs)
    MonthlyDF =reduce(lambda left, right: pd.merge(left, right, on="Dates", how="inner"), Monthly_data_dfs)
    QuarterlyDF = reduce(lambda left, right: pd.merge(left, right, on="Dates", how="inner"), Quarterly_Data_dfs)

    #testing
    #DailyDF.to_excel('dailyEconomicalData.xlsx')
    #MonthlyDF.to_excel('monthlyEconomicalData.xlsx')

    #adding month column to all to merge them on it
    DailyDF["Month"] = DailyDF["Dates"].dt.to_period("M")
    MonthlyDF["Month"] = MonthlyDF["Dates"].dt.to_period("M")

    #merging Day & month df
    Day_MonthDF = pd.merge(DailyDF,MonthlyDF, on="Month", how="left")

    #formating df
    Day_MonthDF = Day_MonthDF.drop(columns=["Month"])
    Day_MonthDF = Day_MonthDF.rename(columns = {"Dates_x": "Dates"})
    Day_MonthDF = Day_MonthDF.drop(columns=["Dates_y"])

    # adding quarter column to merge them on
    Day_MonthDF["Quarter"] = Day_MonthDF["Dates"].dt.to_period("Q")
    QuarterlyDF["Quarter"] = QuarterlyDF["Dates"].dt.to_period("Q")

    #merging day&month df with quarter df
    EconomicalDF = pd.merge(Day_MonthDF, QuarterlyDF,on="Quarter", how="left")

    #formating df
    EconomicalDF = EconomicalDF.drop(columns=["Quarter"])
    EconomicalDF = EconomicalDF.rename(columns={"Dates_x": "Dates"})
    EconomicalDF = EconomicalDF.drop(columns=["Dates_y"])

    print(EconomicalDF.head)
    #testing
    #EconomicalDF.to_excel('AllEconomicalData.xlsx')
    return EconomicalDF


def APICall(Data):# returns array of dfs in form [date, feature]
    #FRED API limits
    ## 120 Requests / minute
    ## 1000 records / request & maybe 100,000 observations / request

    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    All_data = []

    # iterate though econ obj and get df of dates and data and append to All_data
    for obj in Data:
        result = fred.get_series(series_id=obj,observation_start=start, observation_end=end)
        resultdf = pd.DataFrame(result)
        resultdf = resultdf.reset_index()
        resultdf.columns = ["Dates", obj]
        All_data.append(resultdf)
        time.sleep(0.11)

    return All_data

def Merge(HistoricalDF, EconomicalDF):
    MergedDF = pd.merge(HistoricalDF, EconomicalDF,on="Dates", how="inner")

    print(MergedDF.head())
    MergedDF.to_excel('MergedDF.xlsx')  # uses $USD
    return MergedDF


load_dotenv() #load env vars

EconomicDF = GetEconomical()
HistoricDF = GetHistorical()
Merge(EconomicDF, HistoricDF)
