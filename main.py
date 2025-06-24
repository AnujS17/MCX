import newspaper
from newspaper import Article
import nltk
from urllib.parse import urlparse, unquote
import re
from datetime import datetime, timedelta
import feedparser
import pandas as pd
import dateparser
import time
import os

nltk.download('punkt')

def extract_headline_from_url(url):
    path = urlparse(url).path
    last_part = path.rstrip('/').split('/')[-1]
    last_part = re.sub(r'\.htm[l]?$', '', last_part)
    headline = re.sub(r'[-_]', ' ', last_part)
    headline = re.sub(r'^\d+|\d+$', '', headline)
    headline = headline.strip().title()
    headline = unquote(headline)
    return headline if headline else "Unknown Headline"

def get_previous_weekday(start_date, days_back):
    date = start_date
    while days_back > 0:
        date -= timedelta(days=1)
        if date.weekday() < 5:
            days_back -= 1
    return date

def fetch_metalsdaily_news():
    silver_news_url = 'https://www.metalsdaily.com/news/silver-news/'
    paper = newspaper.build(silver_news_url, memoize_articles=False)

    seen_headlines = set()
    rows = []

    for article in paper.articles:
        try:
            article.download()
            article.parse()
            headline = article.title
            if not headline or headline.strip().lower() in [
                "live gold prices", "metalsdaily.com missing page",
                "silver news headlines today", "gold news headlines today"
            ]:
                headline = extract_headline_from_url(article.url)

            if headline not in seen_headlines:
                seen_headlines.add(headline)
                rows.append([headline, ''])  # date to be assigned later
        except Exception as e:
            print(f"Failed to process article: {article.url}")
            print("Error:", e)

    # Simulate dates in blocks of 10 (latest to oldest)
    today = datetime.now()
    date = today
    for row in rows:
        # Find the previous weekday (skip weekends)
        while date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            date -= timedelta(days=1)
        row[1] = date.strftime('%Y-%m-%d')
        date -= timedelta(days=1)


    df = pd.DataFrame(rows, columns=["headline", "date"])
    df["date"] = pd.to_datetime(df["date"])
    return df

def fetch_google_news_rss(query="silver OR silver price OR silver market", max_articles=100):
    query = query.replace(" ", "+")
    feed_url = f"https://news.google.com/rss/search?q={query}+when:30d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(feed_url)

    news_data = []
    for entry in feed.entries[:max_articles]:
        title = entry.title
        published = dateparser.parse(entry.published)
        if title and published:
            news_data.append({
                "headline": title,
                "date": published.date()
            })

    df = pd.DataFrame(news_data)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_existing_data(filename):
    if os.path.exists(filename):
        return pd.read_excel(filename)
    return pd.DataFrame(columns=["headline", "date"])

def deduplicate_and_merge(old_df, new_df):
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["headline"], inplace=True)
    combined["date"] = pd.to_datetime(combined["date"], errors='coerce')
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values(by="date", ascending=False).reset_index(drop=True)
    return combined

def run_scraper():
    print(f"🕒 Running scraper at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    existing_df = load_existing_data("silver_combined_news.xlsx")
    metals_df = fetch_metalsdaily_news()
    google_df = fetch_google_news_rss()
    merged = deduplicate_and_merge(existing_df, pd.concat([metals_df, google_df], ignore_index=True))
    merged.to_excel("silver_combined_news.xlsx", index=False)
    print(f"✅ Updated Excel saved with {len(merged)} unique headlines.\n")

def schedule_every_4_hours():
    while True:
        run_scraper()
        print("🕓 Sleeping for 4 hours...\n")
        time.sleep(4 * 60 * 60)  # Sleep for 4 hours

# Run immediately when the script is started
if __name__ == "__main__":
    schedule_every_4_hours()
