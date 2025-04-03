# AI-Stock-Prediction

DataCollection.py
- Calls  EconomicData.py, HistoricData.py, SentimentalData.py
- Creates a new pandas dataframe called MergedDF from them


EconomicData.py
- Uses Federal Reserve Bank for economic data

HistoricData.py
- Uses Yahoo finance for historical price data

SentimentalData.py
- Uses business insider news titles for sentimental data
- gets 5 news titles for each day (if less than 5 duplicate)
- scores them
- returns DF with scores of 5 articles for each day
