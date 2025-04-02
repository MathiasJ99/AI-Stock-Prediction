from bs4 import BeautifulSoup
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def scrape_page(page, ticker, base_url):
    """Scrapes news articles for a given stock ticker and page number."""
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
            source = article.find('span', class_='latest-news__source').text.strip()
            link = article.find('a', class_='news-link').get('href')
            link = base_url + link if link.startswith('/') else link

            top_sentiment = ''
            sentiment_score = 0

            page_data.append([datetime_raw, title, source, link, top_sentiment, sentiment_score])
        except AttributeError:
            continue

    return page_data


def scrape_news(ticker, base_url, pages, max_workers=10):
    """Scrapes multiple pages of news for a given ticker."""
    columns = ['datetime', 'title', 'source', 'link', 'top_sentiment', 'sentiment_score']
    data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(scrape_page, page, ticker, base_url): page for page in range(1, pages + 1)}

        for future in as_completed(future_to_page):
            result = future.result()
            if result:
                data.extend(result)

    return pd.DataFrame(data, columns=columns)


def process_news_data(df):
    """Processes the scraped news data."""
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

''' 
def give_scores_bert(df):
    # Load a pre-trained RoBERTa model fine-tuned on financial sentiment analysis
    tokenizer = RobertaTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = RobertaForSequenceClassification.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    # Ensure model is in evaluation mode
    model.eval()

    def get_sentiment_score(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

        # Assign a sentiment score (-1 for negative, 0 for neutral, 1 for positive)
        sentiment_score = (-1 * probs[0]) + (0 * probs[1]) + (1 * probs[2])

        return sentiment_score

    # Apply function to each "Title_" column
    for col in df.columns:
        if col.startswith("Title_"):
            df[f"{col}_score"] = df[col].astype(str).apply(get_sentiment_score)

    return df

'''

def cleandf(df): ## what to do with missing values
    title_columns = [col for col in df.columns if col.startswith('Title_')]

    # Calculate the daily average (excluding zeros)
    df['daily_avg'] = df[title_columns].replace(0, pd.NA).mean(axis=1)

    # Replace zeros with the daily average
    for col in title_columns:
        df[col] = df[col].mask(df[col] == 0, df['daily_avg'])

    # Drop the temporary 'daily_avg' column
    df.drop(columns=['daily_avg'], inplace=True)

    print(df.head())
    return df

def GetSenimental():
    ticker = "googl"
    base_url = "https://markets.businessinsider.com"
    pages = 650
    df = scrape_news(ticker, base_url, pages)
    df = process_news_data(df)

    df.to_excel("Sentimental_google_sorted_titles.xlsx", index=False)
    ## loading in news sources
    #file_path = "Sentimental_sorted.xlsx"
    #df = pd.read_excel(file_path)

    df=give_scores_nltk(df)
    #df = give_scores_bert(df)
    df = cleandf(df)


    print(df.head())
    df.to_excel("Sentimental_google_sorted_nltk_scores.xlsx", index=False)
    #df_sorted.to_excel("Sentimental_sorted.xlsx", index=False)
    #print("Data saved to Sentimental_sorted.xlsx")

    return df



#GetSenimental()


