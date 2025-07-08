import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import json
from urllib.parse import urljoin, urlparse
import logging
import feedparser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SilverNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        # Use a more realistic user agent and headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.articles = []
        
    def get_news_api_articles(self, api_key=None):
        """Get articles from News API (requires API key)"""
        if not api_key:
            logger.info("News API key not provided, skipping News API")
            return
            
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'silver OR "precious metals" OR "silver futures" OR "silver price"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            for article in articles:
                try:
                    self.articles.append({
                        'title': article.get('title', 'N/A'),
                        'url': article.get('url', 'N/A'),
                        'source': article.get('source', {}).get('name', 'News API'),
                        'published_date': article.get('publishedAt', 'N/A'),
                        'summary': article.get('description', 'N/A'),
                        'content': article.get('content', 'N/A')[:2000] if article.get('content') else 'N/A'
                    })
                except Exception as e:
                    logger.error(f"Error processing News API article: {e}")
                    continue
                    
            logger.info(f"Scraped {len(articles)} articles from News API")
            
        except Exception as e:
            logger.error(f"Error with News API: {e}")
    
    def get_rss_feed_articles(self, rss_url, source_name, max_articles=20):
        """Generic RSS feed scraper with better error handling"""
        try:
            logger.info(f"Fetching RSS feed from {source_name}...")
            
            # Use custom headers for RSS requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RSS Reader)',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }
            
            # Try to fetch the RSS feed with requests first
            response = self.session.get(rss_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse with feedparser
            feed = feedparser.parse(response.content)
            
            if not feed.entries:
                logger.warning(f"No entries found in RSS feed for {source_name}")
                return
            
            count = 0
            for entry in feed.entries[:max_articles]:
                try:
                    title = entry.title
                    link = entry.link
                    
                    # Filter for silver-related content
                    if self.is_silver_related(title, entry.get('summary', '')):
                        article_data = {
                            'title': title,
                            'url': link,
                            'source': source_name,
                            'published_date': entry.published if hasattr(entry, 'published') else 'N/A',
                            'summary': entry.summary if hasattr(entry, 'summary') else 'N/A',
                            'content': self.extract_content_from_rss(entry)
                        }
                        self.articles.append(article_data)
                        count += 1
                        
                    time.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error processing RSS article from {source_name}: {e}")
                    continue
                    
            logger.info(f"Scraped {count} silver-related articles from {source_name} RSS")
            
        except Exception as e:
            logger.error(f"Error scraping RSS feed {source_name}: {e}")
    
    def extract_content_from_rss(self, entry):
        """Extract content from RSS entry"""
        content = ""
        
        # Try different content fields
        if hasattr(entry, 'content'):
            content = entry.content[0].value if entry.content else ""
        elif hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description
        
        # Clean HTML tags if present
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text(strip=True)
        
        return content[:2000] if content else 'N/A'
    
    def is_silver_related(self, title, summary=""):
        """Check if article is silver-related"""
        silver_keywords = [
            'silver', 'precious metals', 'commodities', 'bullion', 'ag futures',
            'silver price', 'silver market', 'silver mining', 'silver demand',
            'silver supply', 'precious metal', 'gold silver', 'metals trading'
        ]
        
        text = (title + " " + summary).lower()
        return any(keyword in text for keyword in silver_keywords)
    
    def get_alternative_sources(self):
        """Get articles from alternative sources that are more scraping-friendly"""
        
        # List of RSS feeds that are typically more accessible
        rss_feeds = [
            ('https://feeds.finance.yahoo.com/rss/2.0/headline', 'Yahoo Finance'),
            ('https://www.investing.com/rss/news_25.rss', 'Investing.com Commodities'),
            ('https://rss.cnn.com/rss/money_news_international.rss', 'CNN Money'),
            ('https://feeds.bloomberg.com/markets/news.rss', 'Bloomberg Markets'),
            ('https://www.marketwatch.com/rss/topstories', 'MarketWatch Top Stories'),
            ('https://feeds.reuters.com/reuters/businessNews', 'Reuters Business'),
            ('https://www.mining.com/rss/', 'Mining.com'),
        ]
        
        for rss_url, source_name in rss_feeds:
            try:
                logger.info(f"Trying to fetch from {source_name}...")
                self.get_rss_feed_articles(rss_url, source_name, max_articles=15)
                time.sleep(2)  # Be respectful between sources
            except Exception as e:
                logger.error(f"Failed to fetch from {source_name}: {e}")
                continue
    
    def get_google_news_rss(self):
        """Get silver news from Google News RSS"""
        try:
            # Google News RSS for silver-related topics
            queries = [
                'silver price',
                'silver market',
                'precious metals',
                'silver futures',
                'silver mining'
            ]
            
            for query in queries:
                try:
                    # Google News RSS URL
                    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                    
                    response = self.session.get(rss_url, timeout=10)
                    response.raise_for_status()
                    
                    feed = feedparser.parse(response.content)
                    
                    count = 0
                    for entry in feed.entries[:10]:  # Limit per query
                        try:
                            article_data = {
                                'title': entry.title,
                                'url': entry.link,
                                'source': 'Google News',
                                'published_date': entry.published if hasattr(entry, 'published') else 'N/A',
                                'summary': entry.summary if hasattr(entry, 'summary') else 'N/A',
                                'content': 'N/A'  # Google News RSS doesn't include full content
                            }
                            self.articles.append(article_data)
                            count += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing Google News article: {e}")
                            continue
                    
                    logger.info(f"Scraped {count} articles from Google News for query '{query}'")
                    time.sleep(1)  # Rate limiting between queries
                    
                except Exception as e:
                    logger.error(f"Error with Google News query '{query}': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error with Google News RSS: {e}")
    
    def get_free_news_sources(self):
        """Get news from free, accessible sources"""
        try:
            # Try some free financial news APIs and RSS feeds
            free_sources = [
                {
                    'name': 'Alpha Vantage News',
                    'url': 'https://www.alphavantage.co/query',
                    'params': {
                        'function': 'NEWS_SENTIMENT',
                        'tickers': 'SLV,PSLV',  # Silver ETFs
                        'apikey': 'demo'  # Demo key - replace with real key if available
                    }
                }
            ]
            
            # For now, focus on RSS feeds which are more reliable
            logger.info("Focusing on RSS feeds for reliable data...")
            
        except Exception as e:
            logger.error(f"Error with free news sources: {e}")
    
    def scrape_all_sources(self, news_api_key=None):
        """Scrape all available sources"""
        logger.info("Starting to scrape all available sources...")
        
        # Try News API if key provided
        if news_api_key:
            self.get_news_api_articles(news_api_key)
        
        # Get Google News RSS
        self.get_google_news_rss()
        time.sleep(2)
        
        # Get alternative RSS sources
        self.get_alternative_sources()
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in self.articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        self.articles = unique_articles
        logger.info(f"Total unique articles scraped: {len(self.articles)}")
    
    def save_to_excel(self, filename='silver_news_articles.xlsx'):
        """Save scraped articles to Excel file"""
        if not self.articles:
            logger.warning("No articles to save")
            return
        
        try:
            df = pd.DataFrame(self.articles)
            
            # Clean and format data
            df['scraped_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df['title_length'] = df['title'].str.len()
            df['content_length'] = df['content'].str.len()
            
            # Clean up published dates
            df['published_date'] = df['published_date'].astype(str)
            
            # Reorder columns
            columns_order = ['source', 'title', 'published_date', 'url', 'summary', 'content', 
                           'scraped_date', 'title_length', 'content_length']
            df = df.reindex(columns=columns_order)
            
            # Save to Excel with formatting
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Silver News Articles', index=False)
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Silver News Articles']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 80)  # Cap width at 80
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"Articles saved to {filename}")
            print(f"Successfully saved {len(self.articles)} articles to {filename}")
            
            # Print summary by source
            source_counts = df['source'].value_counts()
            print("\nArticles by source:")
            for source, count in source_counts.items():
                print(f"  {source}: {count}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")
            print(f"Error saving to Excel: {e}")

def main():
    """Main function to run the scraper"""
    print("Silver News Scraper")
    print("===================")
    print("This script will collect silver-related news from various RSS feeds and free sources.")
    print("For better results, consider getting a News API key from https://newsapi.org/")
    print()
    
    # Optional: Ask for News API key
    news_api_key = input("Enter your News API key (or press Enter to skip): ").strip()
    if not news_api_key:
        news_api_key = None
        print("Proceeding without News API key...")
    
    scraper = SilverNewsScraper()
    
    try:
        # Scrape all sources
        scraper.scrape_all_sources(news_api_key)
        
        # Save to Excel
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'silver_news_{timestamp}.xlsx'
        scraper.save_to_excel(filename)
        
        # Print summary
        if scraper.articles:
            print(f"\nScraping Summary:")
            print(f"Total articles: {len(scraper.articles)}")
            sources = set(article['source'] for article in scraper.articles)
            print(f"Sources: {', '.join(sources)}")
            print(f"File saved: {filename}")
            
            # Show sample titles
            print(f"\nSample article titles:")
            for i, article in enumerate(scraper.articles[:5]):
                print(f"  {i+1}. {article['title'][:80]}...")
        else:
            print("No articles found. This might be due to:")
            print("1. Network connectivity issues")
            print("2. RSS feeds being temporarily unavailable")
            print("3. All sources blocking requests")
            print("4. No silver-related content in current feeds")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()