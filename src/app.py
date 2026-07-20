# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import uvicorn

# Import our predictor
from predict import WeatherPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Hyderabad Weather Forecast API",
    description="Predict next 7 days of maximum temperature using XGBoost",
    version="1.0.0"
)

# Load the predictor once when the server starts
print("Loading WeatherPredictor...")
predictor = WeatherPredictor()
print("✅ Predictor loaded successfully")

# Define request/response models
class WeatherRequest(BaseModel):
    """Input for prediction"""
    # User can optionally provide their own data
    # For simplicity, we'll use our internal data
    pass

class ForecastResponse(BaseModel):
    """Forecast output"""
    date: str
    temperature_celsius: float

class PredictionResponse(BaseModel):
    """Full prediction response"""
    city: str
    forecast: list[ForecastResponse]
    model_version: str
    generated_at: str

# Health check endpoint
@app.get("/")
@app.get("/health")
def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_weather(request: WeatherRequest):
    """
    Get 7-day temperature forecast for Hyderabad
    """
    try:
        # Load historical data
        df = pd.read_csv('data/processed/daily_weather_features.csv', index_col=0, parse_dates=True)
        
        # Get the last 60 days for prediction (need at least 30 days of data)
        historical = df[['temp_max', 'temp_min', 'temp_avg', 'precip', 'wind_speed', 'humidity']].iloc[-60:]
        
        if len(historical) < 30:
            raise HTTPException(
                status_code=503,
                detail="Not enough historical data. Need at least 30 days."
            )
        
        # Predict next 7 days
        predictions = predictor.predict_days(historical, days=7)
        
        # Format response
        forecast = []
        for i, pred in enumerate(predictions, 1):
            date = (datetime.now().date() + timedelta(days=i)).isoformat()
            forecast.append(ForecastResponse(
                date=date,
                temperature_celsius=round(pred, 1)
            ))
        
        return PredictionResponse(
            city="Hyderabad",
            forecast=forecast,
            model_version="model_latest",
            generated_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint with API info
@app.get("/info")
def get_info():
    """Get model information"""
    return {
        "city": "Hyderabad",
        "coordinates": {"latitude": 17.3850, "longitude": 78.4867},
        "model_type": "XGBoost",
        "features": len(predictor.feature_columns),
        "forecast_days": 7
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )