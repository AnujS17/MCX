import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.svm import SVR
import lightgbm as lgb

# Technical Analysis
import talib

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization and Analysis
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

class SilverTradingStrategy:
    def __init__(self, data_path):
        """
        Initialize the Silver Trading Strategy with sentiment-enhanced features
        """
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.signals = {}
        self.performance_metrics = {}
        
    def load_and_prepare_data(self):
        """
        Load data and perform initial preprocessing
        """
        print("Loading and preparing data...")
        
        # Load main dataset
        self.df = pd.read_excel(self.data_path)
        
        # Convert Date column
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Forward fill missing values for continuous variables
        numeric_cols = ['MCX_Silver_Close_INR', 'MCX_Silver_High_INR', 'MCX_Silver_Low_INR', 
                       'MCX_Silver_Open_INR', 'MCX_Silver_Volume_Lots', 'MCX_Silver_Open_Interest',
                       'Gold_Global_Rate_USD', 'Silver_Global_Rate_USD', 'USD_Index_Value',
                       'Repo Rate (%)', 'Inflation Rate (%)']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(method='ffill')
        
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        return self.df
    
    def engineer_sentiment_features(self):
        """
        Process sentiment data and create aggregated features
        """
        print("Engineering sentiment features...")
        
        # Group by date for sentiment aggregation
        daily_sentiment = self.df.groupby('Date').agg({
            'importance': ['mean', 'sum', 'count'],
            'Headlines': 'count'
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns]
        daily_sentiment = daily_sentiment.reset_index()
        
        # Create placeholder sentiment scores (since we don't have the processed sentiment file)
        # In practice, you would load your sentiment analysis results here
        daily_sentiment['sentiment_score'] = np.random.normal(0, 0.3, len(daily_sentiment))
        daily_sentiment['sentiment_confidence'] = np.random.uniform(0.3, 0.9, len(daily_sentiment))
        daily_sentiment['bullish_signals'] = np.random.poisson(0.5, len(daily_sentiment))
        daily_sentiment['bearish_signals'] = np.random.poisson(0.5, len(daily_sentiment))
        
        # Add derived sentiment features
        daily_sentiment['net_sentiment'] = daily_sentiment['bullish_signals'] - daily_sentiment['bearish_signals']
        daily_sentiment['sentiment_intensity'] = abs(daily_sentiment['sentiment_score'])
        daily_sentiment['news_volume'] = daily_sentiment['Headlines_count']
        
        # Sentiment momentum features
        daily_sentiment['sentiment_sma_3'] = daily_sentiment['sentiment_score'].rolling(3).mean()
        daily_sentiment['sentiment_sma_7'] = daily_sentiment['sentiment_score'].rolling(7).mean()
        daily_sentiment['sentiment_momentum_3'] = daily_sentiment['sentiment_score'] - daily_sentiment['sentiment_sma_3']
        daily_sentiment['sentiment_volatility_7'] = daily_sentiment['sentiment_score'].rolling(7).std()
        
        # Merge with main dataframe
        price_data = self.df.groupby('Date').agg({
            'MCX_Silver_Close_INR': 'last',
            'MCX_Silver_High_INR': 'last',
            'MCX_Silver_Low_INR': 'last',
            'MCX_Silver_Open_INR': 'last',
            'MCX_Silver_Volume_Lots': 'last',
            'MCX_Silver_Open_Interest': 'last',
            'Gold_Global_Rate_USD': 'last',
            'Silver_Global_Rate_USD': 'last',
            'USD_Index_Value': 'last',
            'Repo Rate (%)': 'last',
            'Inflation Rate (%)': 'last'
        }).reset_index()
        
        self.df_processed = pd.merge(price_data, daily_sentiment, on='Date', how='left')
        self.df_processed = self.df_processed.fillna(method='ffill').dropna()
        
        print(f"Sentiment features engineered. Final shape: {self.df_processed.shape}")
        return self.df_processed
    
    def add_technical_indicators(self):
        """
        Add comprehensive technical analysis indicators
        """
        print("Adding technical indicators...")
        
        df = self.df_processed.copy()
        
        # Price-based indicators
        df['returns_1d'] = df['MCX_Silver_Close_INR'].pct_change()
        df['returns_5d'] = df['MCX_Silver_Close_INR'].pct_change(5)
        df['returns_10d'] = df['MCX_Silver_Close_INR'].pct_change(10)
        
        # Volatility measures
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        
        # Moving averages
        df['sma_5'] = df['MCX_Silver_Close_INR'].rolling(5).mean()
        df['sma_10'] = df['MCX_Silver_Close_INR'].rolling(10).mean()
        df['sma_20'] = df['MCX_Silver_Close_INR'].rolling(20).mean()
        df['ema_5'] = df['MCX_Silver_Close_INR'].ewm(span=5).mean()
        df['ema_10'] = df['MCX_Silver_Close_INR'].ewm(span=10).mean()
        
        # Moving average ratios
        df['price_sma5_ratio'] = df['MCX_Silver_Close_INR'] / df['sma_5']
        df['price_sma10_ratio'] = df['MCX_Silver_Close_INR'] / df['sma_10']
        df['sma5_sma20_ratio'] = df['sma_5'] / df['sma_20']
        
        # RSI
        if len(df) > 14:
            df['rsi_14'] = talib.RSI(df['MCX_Silver_Close_INR'].values, timeperiod=14)
        
        # MACD
        if len(df) > 26:
            macd, macd_signal, macd_hist = talib.MACD(df['MCX_Silver_Close_INR'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        if len(df) > 20:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['MCX_Silver_Close_INR'].values)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_position'] = (df['MCX_Silver_Close_INR'] - bb_lower) / (bb_upper - bb_lower)
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Volume indicators
        df['volume_sma_5'] = df['MCX_Silver_Volume_Lots'].rolling(5).mean()
        df['volume_ratio'] = df['MCX_Silver_Volume_Lots'] / df['volume_sma_5']
        
        # Open Interest momentum
        df['oi_change'] = df['MCX_Silver_Open_Interest'].pct_change()
        df['oi_momentum_5'] = df['oi_change'].rolling(5).mean()
        
        # Gold-Silver ratio and USD impact
        df['gold_silver_ratio'] = df['Gold_Global_Rate_USD'] / df['Silver_Global_Rate_USD']
        df['gold_silver_ratio_sma'] = df['gold_silver_ratio'].rolling(10).mean()
        df['gold_silver_ratio_deviation'] = (df['gold_silver_ratio'] - df['gold_silver_ratio_sma']) / df['gold_silver_ratio_sma']
        
        # USD Index momentum
        df['usd_momentum_5'] = df['USD_Index_Value'].pct_change(5)
        df['usd_momentum_10'] = df['USD_Index_Value'].pct_change(10)
        
        # Interest rate differentials
        df['real_interest_rate'] = df['Repo Rate (%)'] - df['Inflation Rate (%)']
        df['rate_change'] = df['Repo Rate (%)'].diff()
        
        self.df_features = df
        print(f"Technical indicators added. Features shape: {self.df_features.shape}")
        return self.df_features
    
    def create_target_variables(self):
        """
        Create prediction targets for different time horizons
        """
        print("Creating target variables...")
        
        df = self.df_features.copy()
        
        # Next day return (classification target)
        df['target_return_1d'] = df['MCX_Silver_Close_INR'].shift(-1) / df['MCX_Silver_Close_INR'] - 1
        
        # Next day direction (binary classification)
        df['target_direction_1d'] = (df['target_return_1d'] > 0).astype(int)
        
        # Multi-day returns
        df['target_return_3d'] = df['MCX_Silver_Close_INR'].shift(-3) / df['MCX_Silver_Close_INR'] - 1
        df['target_return_5d'] = df['MCX_Silver_Close_INR'].shift(-5) / df['MCX_Silver_Close_INR'] - 1
        
        # Volatility target (next 5-day volatility)
        df['target_volatility_5d'] = df['returns_1d'].shift(-5).rolling(5).std()
        
        # Strong movement targets (>2% moves)
        df['target_strong_move'] = (abs(df['target_return_1d']) > 0.02).astype(int)
        
        self.df_targets = df
        print("Target variables created.")
        return self.df_targets
    
    def prepare_ml_datasets(self, train_ratio=0.7, validation_ratio=0.15):
        """
        Prepare training, validation, and test datasets
        """
        print("Preparing ML datasets...")
        
        df = self.df_targets.copy()
        
        # Define feature columns (exclude targets and metadata)
        exclude_cols = ['Date', 'Headlines_count', 'target_return_1d', 'target_direction_1d', 
                       'target_return_3d', 'target_return_5d', 'target_volatility_5d', 'target_strong_move']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining NaN values
        df_clean = df.dropna()
        
        # Split data chronologically
        n = len(df_clean)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        # Create splits
        train_data = df_clean.iloc[:train_end]
        val_data = df_clean.iloc[train_end:val_end]
        test_data = df_clean.iloc[val_end:]
        
        # Prepare features and targets
        X_train = train_data[feature_cols]
        X_val = val_data[feature_cols]
        X_test = test_data[feature_cols]
        
        y_train = train_data['target_return_1d']
        y_val = val_data['target_return_1d']
        y_test = test_data['target_return_1d']
        
        # Binary classification targets
        y_train_dir = train_data['target_direction_1d']
        y_val_dir = val_data['target_direction_1d']
        y_test_dir = test_data['target_direction_1d']
        
        # Store datasets
        self.datasets = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'y_train_dir': y_train_dir, 'y_val_dir': y_val_dir, 'y_test_dir': y_test_dir,
            'train_dates': train_data['Date'],
            'val_dates': val_data['Date'],
            'test_dates': test_data['Date'],
            'train_prices': train_data['MCX_Silver_Close_INR'],
            'val_prices': val_data['MCX_Silver_Close_INR'],
            'test_prices': test_data['MCX_Silver_Close_INR']
        }
        
        self.feature_names = feature_cols
        
        print(f"Dataset prepared:")
        print(f"Training: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples") 
        print(f"Test: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        
        return self.datasets
    
    def scale_features(self):
        """
        Scale features using RobustScaler for outlier resistance
        """
        print("Scaling features...")
        
        # Initialize scaler
        self.scaler = RobustScaler()
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(self.datasets['X_train'])
        X_val_scaled = self.scaler.transform(self.datasets['X_val'])
        X_test_scaled = self.scaler.transform(self.datasets['X_test'])
        
        # Update datasets
        self.datasets['X_train_scaled'] = X_train_scaled
        self.datasets['X_val_scaled'] = X_val_scaled
        self.datasets['X_test_scaled'] = X_test_scaled
        
        print("Feature scaling completed.")
        
    def train_ensemble_models(self):
        """
        Train ensemble of models as recommended in the strategy
        """
        print("Training ensemble models...")
        
        # 1. XGBoost Regressor (40% weight)
        print("Training XGBoost...")
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.models['xgboost'] = xgb.XGBRegressor(**xgb_params)
        self.models['xgboost'].fit(
            self.datasets['X_train_scaled'], 
            self.datasets['y_train'],
            eval_set=[(self.datasets['X_val_scaled'], self.datasets['y_val'])],
            verbose=False
        )

        # 2. LightGBM (35% weight) - Alternative to LSTM for faster training
        print("Training LightGBM...")
        lgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.models['lightgbm'] = lgb.LGBMRegressor(**lgb_params)
        self.models['lightgbm'].fit(
            self.datasets['X_train_scaled'],
            self.datasets['y_train'],
            eval_set=[(self.datasets['X_val_scaled'], self.datasets['y_val'])],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # 3. Support Vector Regression (25% weight)
        print("Training SVR...")
        self.models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.01)
        self.models['svr'].fit(self.datasets['X_train_scaled'], self.datasets['y_train'])
        
        # 4. Random Forest for feature importance
        print("Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100, 
            max_depth=8, 
            random_state=42, 
            n_jobs=-1
        )
        self.models['rf'].fit(self.datasets['X_train_scaled'], self.datasets['y_train'])
        
        print("All models trained successfully.")
        
    def create_ensemble_predictions(self):
        """
        Create weighted ensemble predictions
        """
        print("Creating ensemble predictions...")
        
        # Get individual model predictions
        pred_xgb = self.models['xgboost'].predict(self.datasets['X_test_scaled'])
        pred_lgb = self.models['lightgbm'].predict(self.datasets['X_test_scaled'])
        pred_svr = self.models['svr'].predict(self.datasets['X_test_scaled'])
        
        # Ensemble weights as recommended
        weights = {'xgboost': 0.4, 'lightgbm': 0.35, 'svr': 0.25}
        
        # Create weighted ensemble
        ensemble_pred = (
            weights['xgboost'] * pred_xgb +
            weights['lightgbm'] * pred_lgb +
            weights['svr'] * pred_svr
        )
        
        # Store predictions
        self.predictions = {
            'xgboost': pred_xgb,
            'lightgbm': pred_lgb,
            'svr': pred_svr,
            'ensemble': ensemble_pred,
            'actual': self.datasets['y_test'].values
        }
        
        print("Ensemble predictions created.")
        return self.predictions
        
    def generate_trading_signals(self, confidence_threshold=0.0):
        """
        Generate trading signals with risk management
        """
        print("Generating trading signals...")
        
        pred_returns = self.predictions['ensemble']
        actual_returns = self.predictions['actual']
        
        # Signal generation with confidence thresholds
        signals = []
        positions = []
        
        for i, pred_ret in enumerate(pred_returns):
            if pred_ret > confidence_threshold:
                signal = 1  # Buy signal
            elif pred_ret < -confidence_threshold:
                signal = -1  # Sell signal
            else:
                signal = 0  # Hold
            
            signals.append(signal)
            
        # Position sizing based on Kelly Criterion approximation
        win_rate = np.mean(np.array(signals) * actual_returns > 0)
        avg_win = np.mean(actual_returns[actual_returns > 0]) if len(actual_returns[actual_returns > 0]) > 0 else 0
        avg_loss = np.mean(actual_returns[actual_returns < 0]) if len(actual_returns[actual_returns < 0]) > 0 else 0
        
        if avg_loss != 0:
            kelly_fraction = win_rate - (1 - win_rate) * avg_win / abs(avg_loss)
            kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Cap at 25%
        else:
            kelly_fraction = 0.1
            
        # Apply position sizing
        positions = [signal * kelly_fraction for signal in signals]
        
        self.signals = {
            'signals': np.array(signals),
            'positions': np.array(positions),
            'kelly_fraction': kelly_fraction,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        print(f"Trading signals generated. Kelly fraction: {kelly_fraction:.3f}")
        return self.signals
        
    def backtest_strategy(self):
        """
        Comprehensive backtesting with performance metrics
        """
        print("Running backtest...")
        
        signals = self.signals['signals']
        positions = self.signals['positions']
        actual_returns = self.predictions['actual']
        test_prices = self.datasets['test_prices'].values
        test_dates = self.datasets['test_dates'].values
        
        # Calculate strategy returns
        strategy_returns = positions * actual_returns
        
        # Calculate cumulative returns
        cum_strategy_returns = np.cumprod(1 + strategy_returns) - 1
        cum_market_returns = np.cumprod(1 + actual_returns) - 1
        
        # Performance metrics
        total_return = cum_strategy_returns[-1]
        market_return = cum_market_returns[-1]
        
        # Annualized metrics (assuming daily data)
        trading_days = len(strategy_returns)
        years = trading_days / 252
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annualized_volatility = np.std(strategy_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown analysis
        running_max = np.maximum.accumulate(1 + cum_strategy_returns)
        drawdown = (1 + cum_strategy_returns) / running_max - 1
        max_drawdown = np.min(drawdown)
        
        # Trade analysis
        trades = np.diff(signals) != 0
        num_trades = np.sum(trades)
        
        # Win rate
        winning_trades = strategy_returns > 0
        win_rate = np.mean(winning_trades)
        
        # Profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Store performance metrics
        self.performance_metrics = {
            'total_return': total_return,
            'market_return': market_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'strategy_returns': strategy_returns,
            'cum_strategy_returns': cum_strategy_returns,
            'cum_market_returns': cum_market_returns,
            'dates': test_dates,
            'prices': test_prices,
            'signals': signals,
            'positions': positions
        }
        
        print("Backtest completed.")
        return self.performance_metrics
        
    def print_performance_summary(self):
        """
        Print comprehensive performance summary
        """
        metrics = self.performance_metrics
        
        print("\n" + "="*60)
        print("          SILVER TRADING STRATEGY PERFORMANCE")
        print("="*60)
        
        print(f"\n📊 RETURN METRICS:")
        print(f"   Total Strategy Return:     {metrics['total_return']:.2%}")
        print(f"   Total Market Return:       {metrics['market_return']:.2%}")
        print(f"   Excess Return:             {(metrics['total_return'] - metrics['market_return']):.2%}")
        print(f"   Annualized Return:         {metrics['annualized_return']:.2%}")
        
        print(f"\n📈 RISK METRICS:")
        print(f"   Annualized Volatility:     {metrics['annualized_volatility']:.2%}")
        print(f"   Sharpe Ratio:              {metrics['sharpe_ratio']:.3f}")
        print(f"   Maximum Drawdown:          {metrics['max_drawdown']:.2%}")
        
        print(f"\n🔄 TRADING METRICS:")
        print(f"   Number of Trades:          {metrics['num_trades']}")
        print(f"   Win Rate:                  {metrics['win_rate']:.2%}")
        print(f"   Profit Factor:             {metrics['profit_factor']:.2f}")
        print(f"   Gross Profit:              {metrics['gross_profit']:.4f}")
        print(f"   Gross Loss:                {metrics['gross_loss']:.4f}")
        
        print(f"\n⚙️  MODEL PERFORMANCE:")
        # Calculate prediction accuracy
        pred_direction = np.sign(self.predictions['ensemble'])
        actual_direction = np.sign(self.predictions['actual'])
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        mae = mean_absolute_error(self.predictions['actual'], self.predictions['ensemble'])
        mse = mean_squared_error(self.predictions['actual'], self.predictions['ensemble'])
        r2 = r2_score(self.predictions['actual'], self.predictions['ensemble'])
        
        print(f"   Directional Accuracy:      {directional_accuracy:.2%}")
        print(f"   Mean Absolute Error:       {mae:.4f}")
        print(f"   R² Score:                  {r2:.3f}")
        
        print(f"\n🎯 SIGNAL CHARACTERISTICS:")
        print(f"   Kelly Fraction Used:       {self.signals['kelly_fraction']:.2%}")
        print(f"   Long Signals:              {np.sum(metrics['signals'] == 1)}")
        print(f"   Short Signals:             {np.sum(metrics['signals'] == -1)}")
        print(f"   Hold Periods:              {np.sum(metrics['signals'] == 0)}")
        
        print("="*60)
        
    def plot_results(self):
        """
        Create comprehensive visualization of results
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Cumulative Returns Comparison
        dates = self.performance_metrics['dates']
        axes[0, 0].plot(dates, self.performance_metrics['cum_strategy_returns'], 
                       label='Strategy', linewidth=2, color='blue')
        axes[0, 0].plot(dates, self.performance_metrics['cum_market_returns'], 
                       label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
        axes[0, 0].set_title('Cumulative Returns Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        running_max = np.maximum.accumulate(1 + self.performance_metrics['cum_strategy_returns'])
        drawdown = (1 + self.performance_metrics['cum_strategy_returns']) / running_max - 1
        axes[0, 1].fill_between(dates, drawdown, alpha=0.3, color='red')
        axes[0, 1].plot(dates, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Strategy Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Predictions vs Actual
        axes[1, 0].scatter(self.predictions['actual'], self.predictions['ensemble'], 
                          alpha=0.6, s=20)
        min_val = min(self.predictions['actual'].min(), self.predictions['ensemble'].min())
        max_val = max(self.predictions['actual'].max(), self.predictions['ensemble'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('Actual Returns')
        axes[1, 0].set_ylabel('Predicted Returns')
        axes[1, 0].set_title('Prediction Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance
        if hasattr(self.models['xgboost'], 'feature_importances_'):
            importance = self.models['xgboost'].feature_importances_
            feature_names = self.feature_names
            top_features = sorted(zip(importance, feature_names), reverse=True)[:10]
            
            imp_values, imp_names = zip(*top_features)
            axes[1, 1].barh(range(len(imp_values)), imp_values)
            axes[1, 1].set_yticks(range(len(imp_values)))
            axes[1, 1].set_yticklabels(imp_names, fontsize=8)
            axes[1, 1].set_title('Top 10 Feature Importance (XGBoost)')
        
        # 5. Signal Distribution
        signal_counts = [np.sum(self.performance_metrics['signals'] == -1),
                        np.sum(self.performance_metrics['signals'] == 0),
                        np.sum(self.performance_metrics['signals'] == 1)]
        signal_labels = ['Short', 'Hold', 'Long']
        colors = ['red', 'gray', 'green']
        
        axes[2, 0].pie(signal_counts, labels=signal_labels, colors=colors, autopct='%1.1f%%')
        axes[2, 0].set_title('Signal Distribution')
        
        # 6. Monthly Returns Heatmap
        df_results = pd.DataFrame({
            'Date': dates,
            'Returns': self.performance_metrics['strategy_returns']
        })
        df_results['Date'] = pd.to_datetime(df_results['Date'])
        df_results['Year'] = df_results['Date'].dt.year
        df_results['Month'] = df_results['Date'].dt.month
        
        monthly_returns = df_results.groupby(['Year', 'Month'])['Returns'].sum().unstack(fill_value=0)
        
        if len(monthly_returns) > 0:
            sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', 
                       center=0, ax=axes[2, 1])
            axes[2, 1].set_title('Monthly Returns Heatmap')
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath='silver_trading_model.pkl'):
        """
        Save the trained model and components
        """
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def run_complete_strategy(self):
        """
        Execute the complete trading strategy pipeline
        """
        print("🚀 Starting Silver Trading Strategy Pipeline...")
        print("="*60)
        
        # Data preparation
        self.load_and_prepare_data()
        self.engineer_sentiment_features()
        self.add_technical_indicators()
        self.create_target_variables()
        
        # Model training
        self.prepare_ml_datasets()
        self.scale_features()
        self.train_ensemble_models()
        
        # Strategy execution
        self.create_ensemble_predictions()
        self.generate_trading_signals()
        self.backtest_strategy()
        
        # Results
        self.print_performance_summary()
        self.plot_results()
        
        return self.performance_metrics


# MAIN EXECUTION
if __name__ == "__main__":
    # Initialize and run the strategy
    strategy = SilverTradingStrategy('combined.xlsx')  # Update path as needed
    
    # Run complete pipeline
    results = strategy.run_complete_strategy()
    
    # Save the model for future use
    strategy.save_model('silver_trading_model.pkl')
    
    print("\n✅ Strategy execution completed successfully!")
    print(f"📊 Check the performance summary above for detailed results.")
