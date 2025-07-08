import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# First, install required packages (run this in your terminal/command prompt):
# pip install transformers torch pandas numpy vaderSentiment openpyxl

class SilverSentimentAnalyzer:
    def __init__(self):
        print("Initializing sentiment analyzer...")
        
        # Initialize FinBERT
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            print("✓ FinBERT loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            return None
        
        # Initialize VADER
        self.vader_analyzer = SentimentIntensityAnalyzer()
        print("✓ VADER loaded successfully")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.finbert_model.to(self.device)
        print(f"✓ Using device: {self.device}")
        
        # Enhanced keyword dictionaries for precious metals
        self.bullish_keywords = {
            'inflation': 2.0, 'hedge': 1.8, 'safe-haven': 2.2, 'uncertainty': 1.5,
            'geopolitical': 1.7, 'crisis': 2.0, 'recession': 1.9, 'dovish': 1.6,
            'stimulus': 1.4, 'quantitative easing': 2.1, 'deficit': 1.3, 'tariffs': 1.5,
            'trade war': 1.8, 'currency debasement': 2.3, 'money printing': 2.0,
            'central bank buying': 2.5, 'reserve diversification': 2.2, 'fed pause': 1.4,
            'rate cut': 1.6, 'monetary easing': 1.8, 'dollar weakness': 1.9
        }
        
        self.bearish_keywords = {
            'hawkish': -1.6, 'rate hike': -1.8, 'tightening': -1.5, 'strong dollar': -1.7,
            'yield rise': -1.4, 'economic growth': -1.2, 'risk-on': -1.5, 'equity rally': -1.3,
            'normalization': -1.4, 'tapering': -1.6, 'overvalued': -1.8, 'bubble': -2.0,
            'profit taking': -1.2, 'technical selling': -1.1, 'dollar strength': -1.5
        }
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        
        # Financial abbreviation expansions
        financial_expansions = {
            'fed': 'federal reserve', 'ecb': 'european central bank',
            'boe': 'bank of england', 'rbi': 'reserve bank of india',
            'pmi': 'purchasing managers index', 'cpi': 'consumer price index',
            'gdp': 'gross domestic product', 'fomc': 'federal open market committee',
            'mcx': 'multi commodity exchange', 'comex': 'commodity exchange'
        }
        
        for abbrev, expansion in financial_expansions.items():
            text = re.sub(r'\b' + abbrev + r'\b', expansion, text)
        
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:500]  # Truncate for model limits
    
    def keyword_sentiment_score(self, text):
        """Calculate sentiment based on precious metals keywords"""
        text_lower = text.lower()
        bullish_score = 0
        bearish_score = 0
        
        for keyword, weight in self.bullish_keywords.items():
            if keyword in text_lower:
                bullish_score += weight
        
        for keyword, weight in self.bearish_keywords.items():
            if keyword in text_lower:
                bearish_score += abs(weight)
        
        if bullish_score + bearish_score == 0:
            return 0
        
        net_score = (bullish_score - bearish_score) / (bullish_score + bearish_score)
        return np.clip(net_score, -1, 1)
    
    def get_finbert_sentiment(self, text):
        """Get FinBERT sentiment with error handling"""
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", 
                                          truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()[0]
            
            # Convert to composite score (-1 to +1)
            composite_score = predictions[2] - predictions[0]  # positive - negative
            confidence = float(np.max(predictions))
            
            return composite_score, confidence
            
        except Exception as e:
            print(f"FinBERT error: {e}")
            return 0.0, 0.33
    
    def analyze_sentiment(self, text, importance=1.0):
        """Main sentiment analysis function"""
        try:
            clean_text = self.preprocess_text(text)
            if not clean_text:
                return self._default_result(importance)
            
            # Get FinBERT sentiment
            finbert_score, confidence = self.get_finbert_sentiment(clean_text)
            
            # Get VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(clean_text)
            vader_score = vader_scores['compound']
            
            # Get keyword-based sentiment
            keyword_score = self.keyword_sentiment_score(clean_text)
            
            # Ensemble weighting
            finbert_weight = 0.5
            vader_weight = 0.2
            keyword_weight = 0.3
            
            composite_score = (
                finbert_score * finbert_weight +
                vader_score * vader_weight +
                keyword_score * keyword_weight
            )
            
            # Apply importance weighting
            weighted_score = composite_score * importance
            
            return {
                'sentiment': self._score_to_sentiment(composite_score),
                'confidence': confidence,
                'composite_score': composite_score,
                'weighted_score': weighted_score,
                'finbert_score': finbert_score,
                'vader_score': vader_score,
                'keyword_score': keyword_score,
                'importance': importance,
                'bullish_signal': 1 if weighted_score > 0.15 else 0,
                'bearish_signal': 1 if weighted_score < -0.15 else 0
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return self._default_result(importance)
    
    def _score_to_sentiment(self, score):
        """Convert score to sentiment label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _default_result(self, importance):
        """Default result for error cases"""
        return {
            'sentiment': 'neutral', 'confidence': 0.33, 'composite_score': 0.0,
            'weighted_score': 0.0, 'finbert_score': 0.0, 'vader_score': 0.0,
            'keyword_score': 0.0, 'importance': importance,
            'bullish_signal': 0, 'bearish_signal': 0
        }

def process_news_data(file_path):
    """Main processing function"""
    print(f"Loading data from: {file_path}")
    
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['Headlines', 'Date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Initialize analyzer
        analyzer = SilverSentimentAnalyzer()
        if analyzer is None:
            print("❌ Failed to initialize analyzer")
            return None
        
        print("🔄 Processing sentiment analysis...")
        results = []
        
        for idx, row in df.iterrows():
            # Get importance (default to 1.0 if not present)
            importance = row.get('Importance', 1.0)
            if pd.isna(importance):
                importance = 1.0
            
            # Analyze sentiment
            sentiment_result = analyzer.analyze_sentiment(row['Headlines'], importance)
            
            # Combine with original data
            result = {
                'Date': row['Date'],
                'Headlines': row['Headlines'],
                'Source': row.get('Source', 'Unknown'),
                **sentiment_result
            }
            
            results.append(result)
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(df)} headlines...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create daily aggregation
        print("🔄 Creating daily aggregation...")
        daily_agg = create_daily_aggregation(results_df)
        
        # Save results
        detailed_file = 'detailed_sentiment_analysis.csv'
        daily_file = 'daily_sentiment_aggregated.csv'
        
        results_df.to_csv(detailed_file, index=False)
        daily_agg.to_csv(daily_file, index=False)
        
        print(f"✅ Detailed results saved to: {detailed_file}")
        print(f"✅ Daily aggregated results saved to: {daily_file}")
        
        # Print summary
        print("\n📊 SENTIMENT ANALYSIS SUMMARY:")
        print(f"   Total headlines processed: {len(results_df)}")
        print(f"   Average sentiment score: {results_df['composite_score'].mean():.4f}")
        print(f"   Bullish signals: {results_df['bullish_signal'].sum()}")
        print(f"   Bearish signals: {results_df['bearish_signal'].sum()}")
        print(f"   Date range: {results_df['Date'].min()} to {results_df['Date'].max()}")
        
        return results_df, daily_agg
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def create_daily_aggregation(sentiment_df):
    """Create daily aggregated sentiment features"""
    try:
        # Ensure Date column is datetime
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        
        # Group by date and aggregate
        daily_agg = sentiment_df.groupby('Date').agg({
            'weighted_score': ['mean', 'std', 'min', 'max', 'sum'],
            'composite_score': ['mean', 'std'],
            'confidence': ['mean', 'min'],
            'importance': 'sum',
            'bullish_signal': 'sum',
            'bearish_signal': 'sum',
            'finbert_score': 'mean',
            'vader_score': 'mean',
            'keyword_score': 'mean',
            'Headlines': 'count'
        }).round(4)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
        daily_agg = daily_agg.reset_index()
        
        # Add derived features
        daily_agg['net_sentiment'] = daily_agg['bullish_signal_sum'] - daily_agg['bearish_signal_sum']
        daily_agg['sentiment_intensity'] = abs(daily_agg['weighted_score_mean'])
        daily_agg['news_volume'] = daily_agg['Headlines_count']
        daily_agg['sentiment_range'] = daily_agg['weighted_score_max'] - daily_agg['weighted_score_min']
        
        return daily_agg
        
    except Exception as e:
        print(f"Error in daily aggregation: {e}")
        return pd.DataFrame()

# MAIN EXECUTION
if __name__ == "__main__":
    # UPDATE THIS PATH TO YOUR FILE
    file_path = r"E:\Projects\MCX-Silver\combined.xlsx"  # Change this to your file path
    
    print("🚀 Starting Enhanced Silver Sentiment Analysis")
    print("=" * 50)
    
    # Process the data
    results = process_news_data(file_path)
    
    if results is not None:
        detailed_df, daily_df = results
        print("\n✅ Analysis completed successfully!")
        print("\nOutput files created:")
        print("  1. detailed_sentiment_analysis.csv - Individual headline analysis")
        print("  2. daily_sentiment_aggregated.csv - Daily aggregated features")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
\\\\