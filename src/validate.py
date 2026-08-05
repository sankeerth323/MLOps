# src/validate.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import os

def validate_model():
    """
    Compare new model vs current production model
    Returns True if new model is better
    """
    
    # Load processed data
    df = pd.read_csv('data/processed/daily_weather_features.csv', index_col=0, parse_dates=True)
    
    # Features and target
    target = 'temp_max'
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]
    
    # Split: use last 30 days as test set
    test_size = 30
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    
    # Load scaler
    scaler = joblib.load('models/scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    
    # Load NEW model (just trained)
    new_model = joblib.load('models/model_latest.pkl')
    new_pred = new_model.predict(X_test_scaled)
    
    # Check if OLD model exists (production)
    old_model_path = 'models/model_previous.pkl'
    if os.path.exists(old_model_path):
        old_model = joblib.load(old_model_path)
        old_pred = old_model.predict(X_test_scaled)
        
        # Calculate metrics
        new_mae = mean_absolute_error(y_test, new_pred)
        old_mae = mean_absolute_error(y_test, old_pred)
        
        print(f"New model MAE: {new_mae:.3f}°C")
        print(f"Old model MAE: {old_mae:.3f}°C")
        
        # Decision
        if new_mae < old_mae:
            print("✅ New model is BETTER. Deploying...")
            # Save old model as backup
            os.rename('models/model_latest.pkl', 'models/model_production.pkl')
            return True
        else:
            print("❌ New model is WORSE. Keeping old model.")
            # Revert: delete new model
            os.remove('models/model_latest.pkl')
            return False
    else:
        # No old model exists, so deploy anyway
        print("✅ No previous model. Deploying new model...")
        os.rename('models/model_latest.pkl', 'models/model_production.pkl')
        return True

if __name__ == "__main__":
    validate_model()