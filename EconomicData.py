import time
import pandas as pd
from dotenv import load_dotenv
import os
from fredapi import Fred
from functools import reduce

load_dotenv() #load env vars
start = os.getenv("START")
end = os.getenv("END")


def GetEconomic(file_path):
    if file_path == "EconomicData.xlsx":
        EconomicalDF = pd.read_excel(file_path) ## if already called
        return EconomicalDF
    else:
        ##tags / features
        Daily_Economic_Tags = ["DFF","DFII5","DFII10","T5YIE","T10YIE","VIXCLS","DTWEXBGS","DCOILWTICO","USEPUINDXD"]
        Monthly_Economic_Tags = ["UNRATE","CPIAUCSL","A229RX0","INDPRO","M2SL","PCEPI","UMCSENT","BOPGSTB"]
        Quarterly_Economic_Tags = ["GDP", "A939RX0Q048SBEA","GNP","PSAVE","FGEXPND","CP"]

        # Call API function
        Daily_data_dfs = APICall(Daily_Economic_Tags)
        Monthly_data_dfs = APICall(Monthly_Economic_Tags)
        Quarterly_Data_dfs = APICall(Quarterly_Economic_Tags)

        #Merge data of same freq together on Dates
        DailyDF = reduce(lambda left, right: pd.merge(left, right, on="Dates", how="inner"), Daily_data_dfs)
        MonthlyDF =reduce(lambda left, right: pd.merge(left, right, on="Dates", how="inner"), Monthly_data_dfs)
        QuarterlyDF = reduce(lambda left, right: pd.merge(left, right, on="Dates", how="inner"), Quarterly_Data_dfs)

        #Add month col to Daily and Monthly df, then merge them
        DailyDF["Month"] = DailyDF["Dates"].dt.to_period("M")
        MonthlyDF["Month"] = MonthlyDF["Dates"].dt.to_period("M")
        Day_MonthDF = pd.merge(DailyDF,MonthlyDF, on="Month", how="left")

        #formating df removing unnecessayr cols
        Day_MonthDF = Day_MonthDF.drop(columns=["Month"])
        Day_MonthDF = Day_MonthDF.rename(columns = {"Dates_x": "Dates"})
        Day_MonthDF = Day_MonthDF.drop(columns=["Dates_y"])

        #Add Quarter column  to Day&month and quarter df, then merging them
        Day_MonthDF["Quarter"] = Day_MonthDF["Dates"].dt.to_period("Q")
        QuarterlyDF["Quarter"] = QuarterlyDF["Dates"].dt.to_period("Q")
        EconomicalDF = pd.merge(Day_MonthDF, QuarterlyDF,on="Quarter", how="left")

        #formating df
        EconomicalDF = EconomicalDF.drop(columns=["Quarter"])
        EconomicalDF = EconomicalDF.rename(columns={"Dates_x": "Date"})
        EconomicalDF = EconomicalDF.drop(columns=["Dates_y"])

        EconomicalDF['Date'] = EconomicalDF['Date'].dt.date

        EconomicalDF = EconomicalDF.ffill() ## fix gaps in data
        EconomicalDF = EconomicalDF.bfill()

        print(EconomicalDF.head)
        EconomicalDF.to_excel('EconomicData.xlsx')

    return EconomicalDF


def APICall(Tags):#input: an array of tags, output: array of dfs in form [date, feature]
    #FRED API limits
    ## 120 Requests / minute
    ## 1000 records / request & 100,000 observations / request

    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    data = []

    # iterate though tags provided call, api and get [Dates, data] and add that to data df
    for tag in Tags:
        result = fred.get_series(series_id=tag,observation_start=start, observation_end=end)
        result_df = pd.DataFrame(result)
        result_df = result_df.reset_index()
        result_df.columns = ["Dates", tag]
        data.append(result_df)
        time.sleep(0.11)

    return data
