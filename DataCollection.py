from EconomicData import *
from HistoricData import *
from SentimentalData import *

load_dotenv() #load env vars
start = os.getenv("START")
end = os.getenv("END")

def Merge(HistoricalDF, EconomicalDF, SentimentalDF):
    # make Date column datetime
    HistoricalDF['Date'] = pd.to_datetime(HistoricalDF['Date'])
    EconomicalDF['Date'] = pd.to_datetime(EconomicalDF['Date'])
    SentimentalDF['Date'] = pd.to_datetime(SentimentalDF['Date'])

    HistoricalAndEconomicalDF = pd.merge(HistoricalDF, EconomicalDF, on="Date", how="inner")
    #HistoricalAndEconomicalDF.to_excel("HistoricalAndEconomicalDF.xlsx")
    MergedDF = pd.merge(HistoricalAndEconomicalDF, SentimentalDF, on="Date",  how="left")

    ##Formatting
    MergedDF = MergedDF.drop(columns = ["Unnamed: 0_y", "Unnamed: 0_x"], errors="ignore")

    MergedDF['Date'] = MergedDF['Date'].dt.date
    #fill in missing values
    MergedDF = MergedDF.ffill()
    MergedDF = MergedDF.bfill()

    return MergedDF


def GetData():
    # How to create datasets
    ## GOLD dataset - ticker_his = GC=F, ticker_news = gold
    ## GOOGlE dataset - both = googl

    ticker_historic = "GC=F" ### ticker to uses to get historical data (yahoo finance)
    ticker_news = "gold" ### ticker to use to search for news (https://markets.businessinsider.com/)

    ticker_historic = input("enter stock ticker for yahoo finance: ")
    ticker_news = input("enter stock ticker for news source: ")

    ''' 
    ##MERGE DATA FROM FILE
    HistoricDF = GetHistoric("HistoricalAndTechnicalData_GC=F.xlsx", ticker_historic)
    EconomicDF = GetEconomic("EconomicData.xlsx")
    SentimentalDF = GetSentimental("Sentimental_gold_sorted_scores.xlsx",None)
    '''


    #CALL DATA FROM API AND SCRAPPING
    HistoricDF =  GetHistoric(None, ticker_historic)
    print("fetched historic and technical data")

    EconomicDF = GetEconomic(None)
    print("fetched economic data")

    SentimentalDF = GetSentimental(None, ticker_news)
    print("fetched news data")


    save_file_title = "MergedDF_" + ticker_news + ".xlsx"
    MergedDF = Merge(HistoricDF, EconomicDF,  SentimentalDF)
    MergedDF.to_excel(save_file_title, index=False)

    return MergedDF

#testing
print(GetData().head())

