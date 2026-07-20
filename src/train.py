# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime

def train_model():
    """
    Train XGBoost model to predict tomorrow's max temperature
    """
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load processed data
    input_path = 'data/processed/daily_weather_features.csv'
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    print(f"Loaded data shape: {df.shape}")
    
    # Separate features and target
    # Target: temp_max (predict tomorrow's max temperature)
    # We'll use all other columns as features
    target = 'temp_max'
    features = [col for col in df.columns if col != target]
    
    X = df[features]
    y = df[target]
    
    print(f"Features: {len(features)}")
    print(f"Target: {target}")
    
    # Train/Validation/Test split (chronological!)
    # Use first 70% for train, next 15% for validation, last 15% for test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"\n✅ Scaler saved to models/scaler.pkl")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance on Test Set:")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R² Score: {r2:.4f}")
    
    # Save model with versioning
    version = datetime.now().strftime("%Y_%m_%d")
    model_path = f'models/model_{version}.pkl'
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Also save as 'model_latest.pkl' for easy deployment
    joblib.dump(model, 'models/model_latest.pkl')
    print(f"✅ Model also saved as models/model_latest.pkl")
    
    # Log metrics
    metrics = {
        'date': version,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    # Save metrics to CSV log
    metrics_path = 'models/metrics_log.csv'
    metrics_df = pd.DataFrame([metrics])
    
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Metrics logged to {metrics_path}")
    
    return model, metrics

if __name__ == "__main__":
    train_model()