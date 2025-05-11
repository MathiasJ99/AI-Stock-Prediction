# AI-Stock-Prediction
## Steps to run:
### 1. Run DataCollection.py
### 2. Run nn_optimizer.py

1 Create dataset
-------
DataCollection.py
- asks user for 2 inputs (ticker for historic price and ticker for news price)
  - these will often be the same but in cases where proxy is used for news source e.g barack gold it will be different
- Calls  EconomicData.py, HistoricData.py, SentimentalData.py
- Creates a new pandas dataframe called MergedDF from them
- Saves this new file to MergedDF_{ticker}.xlsx

EconomicData.py
- Uses Federal Reserve Bank API to fetch economic data

HistoricData.py
- Uses Yahoo finance for historical price data

SentimentalData.py
- Uses business insider news titles for sentimental data
- gets 5 news titles for each day (if less than 5 duplicate)
- scores them
- returns DF with scores of 5 articles for each day


2 Run Models
----
nn_optimizer.py:
 - asks user for 3 inputs: 
   - name of study since it uses optuna it records optim process in table and viewable by dashboard
   - choice of model 3 options  (int) 
   - choice of dataset 2 options (int)
 - will then run the study