# src/fetch_data.py
import requests
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_historical_data():
    """
    Fetch historical weather data for Hyderabad from Open-Meteo
    Saves raw data to data/raw/weather_historical.csv
    """
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Hyderabad coordinates
    latitude = 17.3850
    longitude = 78.4867
    
    # Get data for the last 5 years (enough for training)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=5*365)  # 5 years
    
    print(f"Fetching weather data for Hyderabad from {start_date} to {end_date}...")
    
    # Open-Meteo Historical Weather API
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean"
        f"&timezone=Asia/Kolkata"
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' not in data:
            print(f"❌ Error: {data}")
            return None
        
        # Extract daily data
        daily_data = data['daily']
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': daily_data['time'],
            'temp_max': daily_data['temperature_2m_max'],
            'temp_min': daily_data['temperature_2m_min'],
            'temp_avg': daily_data['temperature_2m_mean'],
            'precip': daily_data['precipitation_sum'],
            'wind_speed': daily_data['wind_speed_10m_max'],
            'humidity': daily_data['relative_humidity_2m_mean']
        })
        
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        print(f"✅ Data fetched successfully!")
        print(f"Total records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Print sample data
        print("\nSample data:")
        print(df.head())
        
        # Save raw data to CSV
        output_path = 'data/raw/weather_historical.csv'
        df.to_csv(output_path)
        print(f"\n✅ Data saved to {output_path}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        print("\nPlease check:")
        print("1. Your internet connection")
        print("2. The coordinates are correct")
        return None

if __name__ == "__main__":
    fetch_historical_data()