# src/predict.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

class WeatherPredictor:
    """
    Weather prediction model wrapper
    Loads the latest model and makes predictions
    """
    
    def __init__(self, model_path='models/model_latest.pkl'):
        """
        Initialize the predictor with a trained model
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('models/scaler.pkl')
        self.feature_columns = None
        
        # Load feature columns from training data
        # We need to know which features the model expects
        self._load_feature_columns()
        
        print(f"✅ Predictor initialized with model: {model_path}")
    
    def _load_feature_columns(self):
        """Load feature columns from training data"""
        # Read a sample of processed data to get feature names
        df = pd.read_csv('data/processed/daily_weather_features.csv', index_col=0, parse_dates=True)
        
        # Features are all columns except 'temp_max'
        self.feature_columns = [col for col in df.columns if col != 'temp_max']
        print(f"Loaded {len(self.feature_columns)} features")
    
    def _create_features(self, data):
        """
        Create features for prediction (same as in preprocess.py)
        data: DataFrame with date index and raw weather columns
        """
        # Copy to avoid modifying original
        df = data.copy()
        
        # ---- Temporal features ----
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # ---- Lag features ----
        # We need historical data to create lags
        # For prediction, we need to pass historical data
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'temp_max_lag_{lag}'] = df['temp_max'].shift(lag)
            df[f'temp_min_lag_{lag}'] = df['temp_min'].shift(lag)
            df[f'temp_avg_lag_{lag}'] = df['temp_avg'].shift(lag)
        
        # ---- Rolling statistics ----
        for window in [3, 7, 14]:
            df[f'temp_max_rolling_mean_{window}'] = df['temp_max'].rolling(window).mean()
            df[f'temp_max_rolling_std_{window}'] = df['temp_max'].rolling(window).std()
            df[f'temp_min_rolling_mean_{window}'] = df['temp_min'].rolling(window).mean()
            df[f'temp_avg_rolling_mean_{window}'] = df['temp_avg'].rolling(window).mean()
            df[f'precip_rolling_sum_{window}'] = df['precip'].rolling(window).sum()
        
        # ---- Seasonal features ----
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ---- Interaction features ----
        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['humidity_wind'] = df['humidity'] * df['wind_speed']
        
        # Drop rows with NaN (created from lag/rolling features)
        df = df.dropna()
        
        return df
    
    def predict_next_day(self, historical_df):
        """
        Predict tomorrow's max temperature based on historical data
        
        historical_df: DataFrame with at least 30 days of historical data
                       Must have columns: temp_max, temp_min, temp_avg, 
                                         precip, wind_speed, humidity
        """
        # Ensure we have enough historical data
        if len(historical_df) < 30:
            raise ValueError(f"Need at least 30 days of data. Got {len(historical_df)}")
        
        # Create features
        df = self._create_features(historical_df)
        
        # Get the latest row (today's features)
        latest = df.iloc[-1:].copy()
        
        # Ensure we have all required features
        missing = set(self.feature_columns) - set(latest.columns)
        if missing:
            print(f"Warning: Missing features: {missing}")
            # Add missing columns with default values
            for col in missing:
                latest[col] = 0
        
        # Reorder columns to match training
        latest = latest[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(latest)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction
    
    def predict_days(self, historical_df, days=7):
        """
        Predict max temperature for the next 'days' days
        """
        predictions = []
        df = historical_df.copy()
        
        for i in range(days):
            # Get prediction for next day
            pred = self.predict_next_day(df)
            predictions.append(pred)
            
            # Add prediction to historical data for next iteration
            # (using tomorrow's predicted max and today's other features)
            new_row = pd.DataFrame({
                'temp_max': [pred],
                'temp_min': [df['temp_min'].iloc[-1]],
                'temp_avg': [df['temp_avg'].iloc[-1]],
                'precip': [df['precip'].iloc[-1]],
                'wind_speed': [df['wind_speed'].iloc[-1]],
                'humidity': [df['humidity'].iloc[-1]]
            }, index=[df.index[-1] + timedelta(days=1)])
            
            df = pd.concat([df, new_row])
        
        return predictions

if __name__ == "__main__":
    # Test the predictor
    print("Testing WeatherPredictor...")
    
    # Load historical data
    df = pd.read_csv('data/processed/daily_weather_features.csv', index_col=0, parse_dates=True)
    
    # Get the last 60 days for prediction
    historical = df[['temp_max', 'temp_min', 'temp_avg', 'precip', 'wind_speed', 'humidity']].iloc[-60:]
    
    # Initialize predictor
    predictor = WeatherPredictor()
    
    # Predict next 7 days
    predictions = predictor.predict_days(historical, days=7)
    
    print(f"\n📊 7-Day Forecast for Hyderabad:")
    for i, pred in enumerate(predictions, 1):
        date = datetime.now().date() + timedelta(days=i)
        print(f"  {date}: {pred:.1f}°C")