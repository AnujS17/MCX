import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import random
import pandas as pd
import os
from datetime import datetime

# ================================
# CONFIGURATION SECTION - MODIFY AS NEEDED
# ================================

# List of Twitter accounts to monitor (add/remove as needed)
ACCOUNTS_TO_MONITOR = [
    "KitcoNewsNOW",
    "SilverSeekcom", 
    "Wheaton_PM",
    "PeterSchiff",
    "GoldSilvercom",
    "SilverDoctors",
    "MilesFranklin",
    "SchiffGold"
]

# Keywords to filter tweets (add/remove as needed)
KEYWORDS_TO_TRACK = [
    "silver",
    "commodity", 
    "market",
    "price",
    "precious metals",
    "gold",
    "inflation",
    "mining",
    "bullion",
    "Gold ETF",
    "Silver ETF",
    "investing",
    "war",
    "israel",
    "iran",
    "geopolitical",
    "crisis",
    "tensions",
    "strike",
    "safe haven"
]

# Scraping limits
MAX_TWEETS_PER_ACCOUNT = 100  # Maximum tweets to collect per account
MAX_TOTAL_TWEETS = 1200       # Maximum total tweets to collect
MAX_SCROLLS = 15              # Number of times to scroll down per account

# Output settings
OUTPUT_FILENAME = "silver_tweets.csv"
INCLUDE_TIMESTAMP = True
REMOVE_DUPLICATES = True

# Browser settings
HEADLESS_MODE = False        # Set to True to run browser in background
DELAY_BETWEEN_ACCOUNTS = 6   # Seconds to wait between scraping accounts
SCROLL_DELAY = (2, 5)        # Random delay range between scrolls (min, max)

# ================================
# END CONFIGURATION SECTION
# ================================

class TwitterScraper:
    def __init__(self):
        self.driver = None
        self.collected_tweets = []
        
    def create_driver(self):
        """Create a stealth Chrome driver"""
        options = uc.ChromeOptions()
        
        if HEADLESS_MODE:
            options.add_argument("--headless")
        
        # Stealth options
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = uc.Chrome(options=options)
            return True
        except Exception as e:
            print(f"Error creating driver: {e}")
            return False
    
    def is_relevant_tweet(self, tweet_text):
        """Check if tweet contains relevant keywords"""
        tweet_lower = tweet_text.lower()
        return any(keyword.lower() in tweet_lower for keyword in KEYWORDS_TO_TRACK)
    
    def scrape_account(self, account_name):
        """Scrape tweets from a single account"""
        account_tweets = []
        
        try:
            profile_url = f"https://twitter.com/{account_name}"
            print(f"Scraping @{account_name}...")
            
            self.driver.get(profile_url)
            time.sleep(5)
            
            # Check if login is required
            if "login" in self.driver.current_url.lower():
                print(f"Login required for {account_name}, skipping...")
                return account_tweets
            
            tweets_collected = 0
            
            for scroll in range(MAX_SCROLLS):
                if tweets_collected >= MAX_TWEETS_PER_ACCOUNT:
                    break
                
                print(f"  Scroll {scroll + 1}/{MAX_SCROLLS}")
                
                try:
                    tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                    
                    for tweet_element in tweet_elements:
                        if tweets_collected >= MAX_TWEETS_PER_ACCOUNT:
                            break
                        
                        try:
                            # Extract tweet text
                            text_element = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                            tweet_text = text_element.text
                            
                            # Check if tweet is relevant
                            if self.is_relevant_tweet(tweet_text):
                                # Extract additional info
                                try:
                                    time_element = tweet_element.find_element(By.CSS_SELECTOR, 'time')
                                    timestamp = time_element.get_attribute('datetime')
                                except:
                                    timestamp = datetime.now().isoformat()
                                
                                tweet_data = {
                                    'account': account_name,
                                    'text': tweet_text,
                                    'timestamp': timestamp,
                                    'collected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'relevant_keywords': [kw for kw in KEYWORDS_TO_TRACK if kw.lower() in tweet_text.lower()]
                                }
                                
                                account_tweets.append(tweet_data)
                                tweets_collected += 1
                                print(f"    ✓ Collected tweet {tweets_collected}")
                        
                        except Exception as e:
                            continue
                    
                    # Scroll down
                    scroll_amount = random.randint(600, 1000)
                    self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                    
                    # Random delay
                    delay = random.uniform(SCROLL_DELAY[0], SCROLL_DELAY[1])
                    time.sleep(delay)
                
                except Exception as e:
                    print(f"    Error during scroll: {e}")
                    break
            
            print(f"  Collected {len(account_tweets)} relevant tweets from @{account_name}")
            
        except Exception as e:
            print(f"Error scraping {account_name}: {e}")
        
        return account_tweets
    
    def scrape_all_accounts(self):
        """Scrape all configured accounts"""
        print("Starting Twitter scraping...")
        print("=" * 50)
        print(f"Accounts to monitor: {len(ACCOUNTS_TO_MONITOR)}")
        print(f"Keywords to track: {KEYWORDS_TO_TRACK}")
        print(f"Max tweets per account: {MAX_TWEETS_PER_ACCOUNT}")
        print(f"Max total tweets: {MAX_TOTAL_TWEETS}")
        print("=" * 50)
        
        if not self.create_driver():
            print("Failed to create browser driver")
            return []
        
        total_collected = 0
        
        try:
            for i, account in enumerate(ACCOUNTS_TO_MONITOR):
                if total_collected >= MAX_TOTAL_TWEETS:
                    print(f"Reached maximum total tweets limit ({MAX_TOTAL_TWEETS})")
                    break
                
                print(f"\n[{i+1}/{len(ACCOUNTS_TO_MONITOR)}] Processing @{account}")
                
                account_tweets = self.scrape_account(account)
                self.collected_tweets.extend(account_tweets)
                total_collected += len(account_tweets)
                
                print(f"Total tweets collected so far: {total_collected}")
                
                # Wait between accounts (except for the last one)
                if i < len(ACCOUNTS_TO_MONITOR) - 1:
                    print(f"Waiting {DELAY_BETWEEN_ACCOUNTS} seconds before next account...")
                    time.sleep(DELAY_BETWEEN_ACCOUNTS)
        
        except KeyboardInterrupt:
            print("\nScraping interrupted by user")
        
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
        
        return self.collected_tweets
    
    def save_to_csv(self, tweets):
        """Save collected tweets to CSV file"""
        if not tweets:
            print("No tweets to save")
            return None
        
        df = pd.DataFrame(tweets)
        
        # Remove duplicates if configured
        if REMOVE_DUPLICATES:
            original_count = len(df)
            df = df.drop_duplicates(subset=['text'])
            print(f"Removed {original_count - len(df)} duplicate tweets")
        
        # Save to CSV
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
        print(f"Saved {len(df)} tweets to {OUTPUT_FILENAME}")
        
        return df
    
    def display_summary(self, df):
        """Display summary of collected data"""
        if df is None or df.empty:
            print("No data to display")
            return
        
        print("\n" + "=" * 50)
        print("SCRAPING SUMMARY")
        print("=" * 50)
        print(f"Total tweets collected: {len(df)}")
        print(f"Accounts scraped: {df['account'].nunique()}")
        print(f"Output file: {OUTPUT_FILENAME}")
        
        print("\nTweets per account:")
        account_counts = df['account'].value_counts()
        for account, count in account_counts.items():
            print(f"  @{account}: {count} tweets")
        
        print("\nMost common keywords found:")
        all_keywords = []
        for keywords_list in df['relevant_keywords']:
            all_keywords.extend(keywords_list)
        
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        for keyword, count in keyword_counts.most_common(5):
            print(f"  '{keyword}': {count} times")
        
        print("\nSample tweets:")
        print("-" * 30)
        for i, (_, tweet) in enumerate(df.head(3).iterrows()):
            print(f"{i+1}. @{tweet['account']}: {tweet['text'][:100]}...")
            print(f"   Keywords: {', '.join(tweet['relevant_keywords'])}")
            print()

def main():
    """Main function to run the scraper"""
    scraper = TwitterScraper()
    
    # Display configuration
    print("TWITTER SCRAPER CONFIGURATION")
    print("=" * 40)
    print(f"Accounts to monitor ({len(ACCOUNTS_TO_MONITOR)}):")
    for account in ACCOUNTS_TO_MONITOR:
        print(f"  - @{account}")
    
    print(f"\nKeywords to track ({len(KEYWORDS_TO_TRACK)}):")
    for keyword in KEYWORDS_TO_TRACK:
        print(f"  - '{keyword}'")
    
    print(f"\nLimits:")
    print(f"  - Max tweets per account: {MAX_TWEETS_PER_ACCOUNT}")
    print(f"  - Max total tweets: {MAX_TOTAL_TWEETS}")
    print(f"  - Max scrolls per account: {MAX_SCROLLS}")
    
    print(f"\nOutput: {OUTPUT_FILENAME}")
    print("=" * 40)
    
    # Ask for confirmation
    response = input("\nProceed with scraping? (y/n): ").lower().strip()
    if response != 'y':
        print("Scraping cancelled")
        return
    
    # Run scraper
    tweets = scraper.scrape_all_accounts()
    
    # Save and display results
    df = scraper.save_to_csv(tweets)
    scraper.display_summary(df)
    
    print("\nScraping completed!")

if __name__ == "__main__":
    main()
