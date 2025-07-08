import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class EffectiveSilverStrategy:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.position = 0  # Current position size
        self.position_value = 0  # Current position value
        self.entry_price = 0
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        
        # Strategy parameters
        self.signal_threshold = 0.02  # 2% prediction threshold
        self.stop_loss = 0.03  # 3% stop loss
        self.take_profit = 0.06  # 6% take profit
        self.max_position_size = 0.20  # 20% of capital per trade
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.min_holding_days = 2  # Minimum holding period
        self.max_holding_days = 15  # Maximum holding period
        
        # Risk management
        self.max_daily_loss = 0.02  # 2% daily loss limit
        self.daily_loss = 0
        self.last_trade_date = None
        
    def prepare_data_and_train_model(self, combined_file, sentiment_file):
        """Complete data preparation and model training"""
        print("🔄 Loading and preparing data...")
        
        # Load data
        main_df = pd.read_excel(combined_file)
        sentiment_df = pd.read_csv(sentiment_file)
        
        # Convert dates and merge
        main_df['Date'] = pd.to_datetime(main_df['Date'])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        merged_df = pd.merge(main_df, sentiment_df, on='Date', how='left')
        
        # Fill missing values
        merged_df['weighted_score_mean'] = merged_df['weighted_score_mean'].fillna(0)
        merged_df = merged_df.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Data prepared: {len(merged_df)} rows")
        
        # Create features
        merged_df = self.create_enhanced_features(merged_df)
        
        # Train model and get predictions
        predictions, actuals, dates = self.train_prediction_model(merged_df)
        
        return predictions, actuals, dates, merged_df
    
    def create_enhanced_features(self, df):
        """Create comprehensive feature set"""
        print("🔄 Creating enhanced features...")
        
        # Target variable
        df['Target'] = df['MCX_Silver_Close_INR'].shift(-1)
        
        # Price features
        df['Return_1d'] = df['MCX_Silver_Close_INR'].pct_change()
        df['Return_3d'] = df['MCX_Silver_Close_INR'].pct_change(3)
        df['Return_5d'] = df['MCX_Silver_Close_INR'].pct_change(5)
        
        # Technical indicators
        for window in [5, 10, 20]:
            df[f'SMA_{window}'] = df['MCX_Silver_Close_INR'].rolling(window).mean()
            df[f'Price_vs_SMA_{window}'] = df['MCX_Silver_Close_INR'] / df[f'SMA_{window}'] - 1
            df[f'Volatility_{window}'] = df['Return_1d'].rolling(window).std()
        
        # Volume indicators
        df['Volume_SMA_5'] = df['MCX_Silver_Volume_Lots'].rolling(5).mean()
        df['Volume_Ratio'] = df['MCX_Silver_Volume_Lots'] / df['Volume_SMA_5']
        
        # Market structure
        df['High_Low_Range'] = (df['MCX_Silver_High_INR'] - df['MCX_Silver_Low_INR']) / df['MCX_Silver_Close_INR']
        df['Close_vs_High'] = (df['MCX_Silver_Close_INR'] - df['MCX_Silver_Low_INR']) / (df['MCX_Silver_High_INR'] - df['MCX_Silver_Low_INR'])
        
        # Global factors
        df['Gold_Silver_Ratio'] = df['Gold_Global_Rate_USD'] / df['Silver_Global_Rate_USD']
        df['USD_Change'] = df['USD_Index_Value'].pct_change()
        df['Gold_Change'] = df['Gold_Global_Rate_USD'].pct_change()
        
        # Macro factors
        df['Real_Rate'] = df['Repo Rate (%)'] - df['Inflation Rate (%)']
        
        # Sentiment features
        df['Sentiment_MA_3'] = df['weighted_score_mean'].rolling(3).mean()
        df['Sentiment_MA_7'] = df['weighted_score_mean'].rolling(7).mean()
        df['Sentiment_Momentum'] = df['weighted_score_mean'] - df['Sentiment_MA_7']
        
        # Interaction features
        df['Volume_Price_Momentum'] = df['Volume_Ratio'] * df['Return_1d']
        df['Sentiment_Price_Interaction'] = df['weighted_score_mean'] * df['Return_1d']
        
        print(f"✓ Enhanced features created: {len(df.columns)} columns")
        return df
    
    def train_prediction_model(self, df):
        """Train Ridge model with proper validation"""
        print("🔄 Training prediction model...")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in 
                       ['Date', 'Headlines', 'Source', 'Target', 'MCX_Silver_Settlement_INR', 'MCX_Silver_Spot_INR']]
        
        # Clean data
        df_clean = df.dropna(subset=['Target']).copy()
        X = df_clean[feature_cols].fillna(method='ffill').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        y = df_clean['Target']
        dates = df_clean['Date']
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(25, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        for train_idx, test_idx in tscv.split(X_selected):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale and train
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            predictions = model.predict(X_test_scaled)
            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            all_dates.extend(dates.iloc[test_idx].values)
        
        print(f"✓ Model trained with {len(all_predictions)} predictions")
        return all_predictions, all_actuals, all_dates
    
    def generate_trading_signals(self, predictions, actuals, dates, sentiment_scores):
        """Generate high-quality trading signals"""
        print("🔄 Generating trading signals...")
        
        signals = []
        signal_strength = []
        
        for i in range(1, len(predictions)):
            # Calculate predicted return
            pred_return = (predictions[i] - actuals[i-1]) / actuals[i-1]
            
            # Sentiment confirmation
            sentiment = sentiment_scores[i] if i < len(sentiment_scores) else 0
            sentiment_confirmation = True
            
            if pred_return > 0 and sentiment < -0.3:  # Don't buy with very negative sentiment
                sentiment_confirmation = False
            elif pred_return < 0 and sentiment > 0.3:  # Don't sell with very positive sentiment
                sentiment_confirmation = False
            
            # Volatility filter
            recent_returns = [actuals[j]/actuals[j-1] - 1 for j in range(max(1, i-5), i)]
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.02
            
            # Adjust threshold based on volatility
            dynamic_threshold = self.signal_threshold * (1 + volatility * 5)
            
            # Generate signal
            if abs(pred_return) > dynamic_threshold and sentiment_confirmation:
                if pred_return > 0:
                    signal = 1  # Buy
                else:
                    signal = -1  # Sell
                strength = min(1.0, abs(pred_return) / dynamic_threshold)
            else:
                signal = 0
                strength = 0
            
            signals.append(signal)
            signal_strength.append(strength)
        
        # Add initial signal
        signals = [0] + signals
        signal_strength = [0] + signal_strength
        
        print(f"✓ Generated {sum([1 for s in signals if s != 0])} trading signals")
        return signals, signal_strength
    
    def execute_strategy(self, predictions, actuals, dates, sentiment_scores):
        """Execute the complete trading strategy"""
        print("🔄 Executing trading strategy...")
        
        # Generate signals
        signals, signal_strength = self.generate_trading_signals(
            predictions, actuals, dates, sentiment_scores
        )
        
        # Execute trades
        for i in range(len(signals)):
            current_date = pd.to_datetime(dates[i])
            current_price = actuals[i]
            signal = signals[i]
            strength = signal_strength[i]
            
            # Update daily loss tracking
            if self.last_trade_date is None or current_date.date() != self.last_trade_date.date():
                self.daily_loss = 0
            
            # Check if we can trade today
            if self.daily_loss >= self.max_daily_loss:
                signal = 0  # No trading if daily loss limit reached
            
            # Process existing position
            if self.position != 0:
                self.check_exit_conditions(current_price, current_date, i)
            
            # Enter new position
            if self.position == 0 and signal != 0:
                self.enter_position(signal, strength, current_price, current_date)
            
            # Update portfolio value
            self.update_portfolio_value(current_price, current_date)
        
        # Close final position
        if self.position != 0:
            self.close_position(actuals[-1], dates[-1], "Strategy End")
        
        return self.calculate_performance()
    
    def enter_position(self, signal, strength, price, date):
        """Enter a new position"""
        # Calculate position size
        position_value = self.cash * self.max_position_size * strength
        position_size = position_value / price
        transaction_cost = position_value * self.transaction_cost
        
        # Check if we have enough cash
        if position_value + transaction_cost > self.cash:
            return
        
        # Execute trade
        self.position = position_size * signal  # Positive for long, negative for short
        self.position_value = position_value
        self.entry_price = price
        self.cash -= (position_value + transaction_cost)
        self.last_trade_date = pd.to_datetime(date)
        
        # Record trade
        trade = {
            'date': date,
            'type': 'BUY' if signal > 0 else 'SELL',
            'price': price,
            'quantity': abs(self.position),
            'value': position_value,
            'transaction_cost': transaction_cost,
            'status': 'OPEN'
        }
        self.trades.append(trade)
        
        print(f"{date}: {trade['type']} {trade['quantity']:.2f} units at ₹{price:.2f}")
    
    def check_exit_conditions(self, current_price, current_date, day_index):
        """Check if we should exit current position"""
        if self.position == 0:
            return
        
        # Calculate current P&L
        if self.position > 0:  # Long position
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Days in position
        days_held = (pd.to_datetime(current_date) - self.last_trade_date).days
        
        # Exit conditions
        exit_reason = None
        
        if pnl_pct <= -self.stop_loss:
            exit_reason = "Stop Loss"
        elif pnl_pct >= self.take_profit:
            exit_reason = "Take Profit"
        elif days_held >= self.max_holding_days:
            exit_reason = "Max Holding Period"
        elif days_held >= self.min_holding_days and abs(pnl_pct) < 0.005:
            exit_reason = "Minimal Movement"
        
        if exit_reason:
            self.close_position(current_price, current_date, exit_reason)
    
    def close_position(self, price, date, reason):
        """Close current position"""
        if self.position == 0:
            return
        
        # Calculate P&L
        position_value = abs(self.position) * price
        transaction_cost = position_value * self.transaction_cost
        
        if self.position > 0:  # Long position
            pnl = (price - self.entry_price) * self.position - transaction_cost
        else:  # Short position
            pnl = (self.entry_price - price) * abs(self.position) - transaction_cost
        
        # Update cash
        self.cash += position_value + pnl
        
        # Update daily loss
        if pnl < 0:
            self.daily_loss += abs(pnl) / self.initial_capital
        
        # Update last trade
        if self.trades:
            self.trades[-1].update({
                'exit_date': date,
                'exit_price': price,
                'exit_reason': reason,
                'pnl': pnl,
                'return': pnl / self.position_value,
                'status': 'CLOSED'
            })
        
        print(f"{date}: CLOSE at ₹{price:.2f} ({reason}) - P&L: ₹{pnl:.2f}")
        
        # Reset position
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
    
    def update_portfolio_value(self, price, date):
        """Update portfolio value"""
        portfolio_value = self.cash
        
        if self.position != 0:
            position_market_value = abs(self.position) * price
            if self.position > 0:
                unrealized_pnl = (price - self.entry_price) * self.position
            else:
                unrealized_pnl = (self.entry_price - price) * abs(self.position)
            portfolio_value += unrealized_pnl
        
        self.portfolio_values.append(portfolio_value)
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)
    
    def calculate_performance(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {'error': 'No closed trades'}
        
        # Basic metrics
        total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Trade metrics
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(closed_trades)
        avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        if len(self.daily_returns) > 1:
            volatility = np.std(self.daily_returns) * np.sqrt(252)
            sharpe_ratio = (total_return * 252) / (volatility * 252) if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252/len(self.portfolio_values)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'final_capital': self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        }

def run_effective_silver_strategy(combined_file, sentiment_file):
    """Run the effective silver trading strategy"""
    
    strategy = EffectiveSilverStrategy(initial_capital=1000000)
    
    # Prepare data and train model
    predictions, actuals, dates, merged_df = strategy.prepare_data_and_train_model(
        combined_file, sentiment_file
    )
    
    # Get sentiment scores
    sentiment_scores = []
    for date in dates:
        matching_sentiment = merged_df[merged_df['Date'] == pd.to_datetime(date)]
        if not matching_sentiment.empty and 'weighted_score_mean' in matching_sentiment.columns:
            sentiment_scores.append(matching_sentiment['weighted_score_mean'].iloc[0])
        else:
            sentiment_scores.append(0)
    
    # Execute strategy
    performance = strategy.execute_strategy(predictions, actuals, dates, sentiment_scores)
    
    if 'error' in performance:
        print(f"❌ Strategy failed: {performance['error']}")
        return None, None
    
    # Print results
    print("\n" + "="*50)
    print("EFFECTIVE STRATEGY PERFORMANCE")
    print("="*50)
    print(f"Initial Capital: ₹{strategy.initial_capital:,.2f}")
    print(f"Final Capital: ₹{performance['final_capital']:,.2f}")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Volatility: {performance['volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {performance['max_drawdown']:.2%}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Average Win: {performance['avg_win']:.2%}")
    print(f"Average Loss: {performance['avg_loss']:.2%}")
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Winning Trades: {performance['winning_trades']}")
    print(f"Losing Trades: {performance['losing_trades']}")
    
    return strategy, performance

# Usage
if __name__ == "__main__":
    combined_file = "combined.xlsx"
    sentiment_file = "daily_sentiment_aggregated.csv"
    
    strategy, performance = run_effective_silver_strategy(combined_file, sentiment_file)
