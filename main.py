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

EXCEL_PATH = r"E:\Projects\MCX-Silver\silver_combined_news1114.xlsx"

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

def shift_weekend_to_monday(date):
    """
    Shift weekend dates (Saturday and Sunday) to the following Monday
    """
    if date.weekday() == 5:  # Saturday
        return date + timedelta(days=2)  # Move to Monday
    elif date.weekday() == 6:  # Sunday
        return date + timedelta(days=1)  # Move to Monday
    else:
        return date  # Weekday, no change needed

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
                rows.append([headline, '', 'MetalsDaily'])  # Added source column
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
        
        # Apply weekend shift logic
        adjusted_date = shift_weekend_to_monday(date)
        row[1] = adjusted_date.strftime('%Y-%m-%d')
        date -= timedelta(days=1)

    df = pd.DataFrame(rows, columns=["headline", "date", "source"])
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
        
        # Extract source from the title (Google News format usually includes source)
        source = "Google News"
        if " - " in title:
            # Google News titles often end with " - Source Name"
            parts = title.split(" - ")
            if len(parts) > 1:
                source = parts[-1].strip()
                title = " - ".join(parts[:-1]).strip()  # Remove source from title
        
        if title and published:
            # Apply weekend shift logic to Google news dates
            adjusted_date = shift_weekend_to_monday(published)
            news_data.append({
                "headline": title,
                "date": adjusted_date.date(),
                "source": source
            })

    df = pd.DataFrame(news_data)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_existing_data(filename):
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        # Ensure source column exists in existing data
        if 'source' not in existing_df.columns:
            existing_df['source'] = 'Unknown'  # Default source for existing entries
        # Remove importance column if it exists
        if 'importance' in existing_df.columns:
            existing_df = existing_df.drop(columns=['importance'])
        return existing_df
    return pd.DataFrame(columns=["headline", "date", "source"])

def deduplicate_and_merge(old_df, new_df):
    # Ensure both dataframes have the source column
    if 'source' not in old_df.columns:
        old_df['source'] = 'Unknown'
    if 'source' not in new_df.columns:
        new_df['source'] = 'Unknown'
    
    # Remove importance column if it exists in either dataframe
    if 'importance' in old_df.columns:
        old_df = old_df.drop(columns=['importance'])
    if 'importance' in new_df.columns:
        new_df = new_df.drop(columns=['importance'])
    
    # Apply weekend shift only to NEW data before merging
    new_df["date"] = pd.to_datetime(new_df["date"], errors='coerce')
    new_df = new_df.dropna(subset=["date"])
    new_df["date"] = new_df["date"].apply(shift_weekend_to_monday)
    
    # Ensure old data has proper datetime format but don't modify dates
    old_df["date"] = pd.to_datetime(old_df["date"], errors='coerce')
    old_df = old_df.dropna(subset=["date"])
    
    combined = pd.concat([old_df, new_df], ignore_index=True)
    
    # Remove duplicates based on headline, keeping the first occurrence
    # This preserves existing sources for duplicates
    combined.drop_duplicates(subset=["headline"], keep='first', inplace=True)
    
    combined = combined.sort_values(by="date", ascending=False).reset_index(drop=True)
    return combined

def run_scraper():
    print(f"🕒 Running scraper at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    existing_df = load_existing_data(EXCEL_PATH)
    metals_df = fetch_metalsdaily_news()
    google_df = fetch_google_news_rss()
    
    # Merge all data, including any existing entries in the file
    new_data = pd.concat([metals_df, google_df], ignore_index=True)
    merged = deduplicate_and_merge(existing_df, new_data)
    
    merged.to_excel(EXCEL_PATH, index=False)
    print(f"✅ Updated Excel saved with {len(merged)} unique headlines.\n")
    
    # Show source statistics
    source_counts = merged['source'].value_counts()
    print("📊 Headlines by source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} headlines")

def schedule_every_4_hours():
    while True:
        run_scraper()
        print("🕓 Sleeping for 4 hours...\n")
        time.sleep(4 * 60 * 60)  # Sleep for 4 hours

# Run immediately when the script is started
if __name__ == "__main__":
    schedule_every_4_hours()
