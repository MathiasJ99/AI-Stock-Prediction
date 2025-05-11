from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def scrape_page(page, ticker, base_url):
    #Scrapes news articles for a given stock ticker and page number
    url = f'{base_url}/news/{ticker}-stock?p={page}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'lxml')
    articles = soup.find_all('div', class_='latest-news__story')

    page_data = []
    for article in articles:
        try:
            datetime_raw = article.find('time', class_='latest-news__date').get('datetime')
            title = article.find('a', class_='news-link').text.strip()
            page_data.append([datetime_raw, title])

        except AttributeError:
            continue

    return page_data


def scrape_news(ticker, base_url, pages, max_workers=10):
    #Scrapes multiple pages of news for a given ticker.
    columns = ['datetime', 'title']
    data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(scrape_page, page, ticker, base_url): page for page in range(1, pages + 1)}
        for future in as_completed(future_to_page):
            result = future.result()
            if result:
                data.extend(result)

    return pd.DataFrame(data, columns=columns)


def process_news_data(df):
    #Processes the scraped news data.
    if df.empty:
        print("No data scraped.")
        return None

    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['Date'] = df['datetime'].dt.date

    df = df.sort_values('datetime', ascending=False)

    df['article_number'] = df.groupby('Date').cumcount() + 1
    df_filtered = df[df['article_number'] <= 5]

    df_sorted = df_filtered.pivot(index='Date', columns='article_number', values='title')
    df_sorted.columns = [f'Title_{int(col)}' for col in df_sorted.columns]

    df_sorted = df_sorted.reset_index()

    print(f'{len(df)} headlines scraped')
    return df_sorted


def give_scores_nltk(df):
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

    for col in df.columns:
        if col.startswith("Title_"):
            df[col] = df[col].astype(str).apply(lambda text: sia.polarity_scores(text)['compound'])

    return df


def give_scores_bert(df):
    #load pre-trained RoBERTa model fine-tuned on financial sentiment analysis
    tokenizer = RobertaTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = RobertaForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


    model.eval()
    def get_sentiment_score(title):
        inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        #convert to prob
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
        #assign sentiment score (-1 for negative, 0 for neutral, 1 for positive)
        sentiment_score = (-1 * probs[0]) + (0 * probs[1]) + (1 * probs[2])
        return sentiment_score

    # Apply func to each "Title_" column
    for col in df.columns:
        if col.startswith("Title_"):
            df[col] = df[col].astype(str).apply(get_sentiment_score)

    return df

def cleandf(df): ## what to do with missing values
    title_columns = [col for col in df.columns if col.startswith('Title_')]

    # Calculate the daily average (excluding zeros)
    df['daily_avg'] = df[title_columns].replace(0, pd.NA).mean(axis=1)

    # if less than 5 titles, replace missing scores with daily average
    for col in title_columns:
        df[col] = df[col].mask(df[col] == 0, df['daily_avg'])

    # Drop the temporary 'daily_avg' column
    df.drop(columns=['daily_avg'], inplace=True)

    df = df.ffill()## fill any gaps

    print(df.head())
    return df

def GetSentimental(file_name, ticker):
    sentimental_path = "Sentimental_" + ticker + "_sorted_scores.xlsx"

    if (file_name == sentimental_path):
        df = pd.read_excel(file_name)
        return df

    else:
        ##options
        if ticker == None:
            ticker = "googl"
        base_url = "https://markets.businessinsider.com"
        pages = 650

        #gather data and save it to excel file
        df = scrape_news(ticker, base_url, pages)
        df = process_news_data(df)
        file_path_titles = "Sentimental_"+ticker+"_sorted_titles.xlsx"
        df.to_excel(file_path_titles, index=False)

        #score data (REMOVE 1)  and clean data
        #df=give_scores_nltk(df)
        print("scoring news data")
        df=give_scores_bert(df)
        df=cleandf(df)

        #output and saving data
        file_path_scores = "Sentimental_"+ticker+"_sorted_scores.xlsx"
        print(df.head())
        df.to_excel(file_path_scores, index=False)

        return df



#GetSentimental(None,None)


