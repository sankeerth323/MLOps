import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.predict import WeatherPredictor

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Hyderabad Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card h4 {
        color: #6c757d;
        font-size: 0.85rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        color: #212529;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0.3rem 0 0 0;
    }
    .forecast-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .forecast-card h4 {
        color: rgba(255,255,255,0.85);
        font-size: 0.8rem;
        margin: 0;
    }
    .forecast-card p {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0 0 0;
    }
    .forecast-card .date {
        color: rgba(255,255,255,0.8);
        font-size: 0.75rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102,126,234,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="main-header">
    <h1>🌤️ Hyderabad Weather Forecast</h1>
    <p>AI-powered 7-day maximum temperature prediction</p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_predictor():
    return WeatherPredictor(model_path='models/model_production.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/daily_weather_features.csv', index_col=0, parse_dates=True)

@st.cache_data
def load_metrics():
    try:
        return pd.read_csv('models/metrics_log.csv')
    except:
        return None

try:
    predictor = load_predictor()
    df = load_data()
    metrics = load_metrics()
    historical = df[['temp_max', 'temp_min', 'temp_avg', 'precip', 'wind_speed', 'humidity']].iloc[-60:]
except Exception as e:
    st.error(f"❌ Initialization failed: {e}")
    st.stop()

# ---------- MODEL METRICS ROW ----------
if metrics is not None and len(metrics) > 0:
    latest = metrics.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>MAE</h4>
            <p>{latest['mae']:.2f}°C</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>RMSE</h4>
            <p>{latest['rmse']:.2f}°C</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>R² Score</h4>
            <p>{latest['r2']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Last Trained</h4>
            <p>{latest['date']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- FORECAST BUTTON ----------
if st.button("🔮 Generate 7-Day Forecast", type="primary"):
    with st.spinner("Running inference..."):
        try:
            predictions = predictor.predict_days(historical, days=7)
            
            dates = [(datetime.now().date() + timedelta(days=i)) for i in range(1, 8)]
            temps = [round(float(p), 3) for p in predictions]
            
            # Forecast cards
            st.markdown("### 📅 7-Day Forecast")
            cols = st.columns(7)
            for i, (date, temp) in enumerate(zip(dates, temps)):
                with cols[i]:
                    st.markdown(f"""
                    <div class="forecast-card">
                        <h4>{date.strftime('%a')}</h4>
                        <p>{temp:.3f}°</p>
                        <div class="date">{date.strftime('%b %d')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interactive chart
            st.markdown("### 📈 Temperature Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[d.strftime('%b %d') for d in dates],
                y=temps,
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10, color='#764ba2'),
                hovertemplate='<b>%{x}</b><br>Max Temp: %{y:.3f}°C<extra></extra>'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Max Temperature (°C)",
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.markdown("### 📊 Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average", f"{np.mean(temps):.3f}°C")
            with col2:
                st.metric("Highest", f"{max(temps):.3f}°C", delta=f"{max(temps) - temps[0]:+.3f}°C")
            with col3:
                st.metric("Lowest", f"{min(temps):.3f}°C", delta=f"{min(temps) - temps[0]:+.3f}°C")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")