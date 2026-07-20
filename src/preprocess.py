# src/preprocess.py
import pandas as pd
import numpy as np
import os

def preprocess_data():
    """
    Load raw weather data, clean it, engineer features, and save processed data
    """
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Load raw data
    input_path = 'data/raw/weather_historical.csv'
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 1: Handle missing values
    print("\nChecking for missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found: {missing[missing > 0].to_dict()}")
        # Forward fill (use previous day's value)
        df = df.fillna(method='ffill')
        # If still missing, use backward fill
        df = df.fillna(method='bfill')
        print("Missing values filled using forward/backward fill")
    else:
        print("✅ No missing values found")
    
    # Step 2: Create features
    print("\nEngineering features...")
    
    # ---- Temporal features ----
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # ---- Lag features (past temperatures) ----
    # Use target = temp_max
    for lag in [1, 2, 3, 7, 14, 30]:  # days
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
    # Sin/Cos encoding for month to capture seasonal cycles
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ---- Interaction features ----
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['humidity_wind'] = df['humidity'] * df['wind_speed']
    
    # Step 3: Drop rows with NaN (created from lag/rolling features)
    df = df.dropna()
    
    print(f"Shape after feature engineering: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    print(f"Features: {list(df.columns)}")
    
    # Step 4: Save processed data
    output_path = 'data/processed/daily_weather_features.csv'
    df.to_csv(output_path)
    print(f"\n✅ Processed data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    preprocess_data()