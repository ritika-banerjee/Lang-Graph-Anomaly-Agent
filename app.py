# streamlit_app.py

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import time
from collections import deque
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("📈 Real-Time Anomaly Detection Dashboard")

# Sidebar configuration
st.sidebar.header("Configuration")
INTERVAL = st.sidebar.slider("Update Interval (seconds)", 0.1, 5.0, 1.0, 0.1)
SEQ_LEN = st.sidebar.number_input("Sequence Length", 10, 100, 60)
API_ENDPOINT = st.sidebar.text_input("API Endpoint", "http://localhost:8000/anomaly_check")

# Load static dataset
@st.cache_data
def load_data():
    df = pd.read_csv("archive/MetroPT3(AirCompressor).csv")
    df = df.drop(columns=["timestamp"], errors="ignore")
    return df

try:
    df = load_data()
    st.sidebar.success(f"Dataset loaded: {len(df)} rows")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

feature_columns = [
    'TP2', 'TP3', 'H1', 'DV_pressure',
    'Reservoirs', 'Oil_temperature', 'Motor_current'
]

# Verify columns exist
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

# Initialize session state for persistence
if 'buffer' not in st.session_state:
    st.session_state.buffer = deque(maxlen=SEQ_LEN)
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'metric_history' not in st.session_state:
    st.session_state.metric_history = []
if 'anomaly_flags' not in st.session_state:
    st.session_state.anomaly_flags = []
if 'reconstruction_losses' not in st.session_state:
    st.session_state.reconstruction_losses = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Control buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Streaming"):
        st.session_state.is_running = True
with col2:
    if st.button("⏸️ Pause"):
        st.session_state.is_running = False
with col3:
    if st.button("🔄 Reset"):
        st.session_state.buffer.clear()
        st.session_state.timestamps = []
        st.session_state.metric_history = []
        st.session_state.anomaly_flags = []
        st.session_state.reconstruction_losses = []
        st.session_state.current_index = 0
        st.session_state.is_running = False

# Metrics display
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
with metrics_col1:
    total_points = len(st.session_state.timestamps)
    st.metric("Total Points", total_points)
with metrics_col2:
    total_anomalies = sum(st.session_state.anomaly_flags)
    st.metric("Anomalies Detected", total_anomalies)
with metrics_col3:
    if total_points > 0:
        anomaly_rate = (total_anomalies / total_points) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    else:
        st.metric("Anomaly Rate", "0%")
with metrics_col4:
    if st.session_state.reconstruction_losses:
        avg_loss = np.mean(st.session_state.reconstruction_losses)
        st.metric("Avg Loss", f"{avg_loss:.4f}")
    else:
        st.metric("Avg Loss", "N/A")

# Chart placeholders
chart_placeholder = st.empty()
status_placeholder = st.empty()

# Function to update charts
def update_charts():
    if not st.session_state.timestamps:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Motor Current with Anomalies', 'Reconstruction Loss'),
        vertical_spacing=0.1
    )
    
    # Motor current plot
    fig.add_trace(
        go.Scatter(
            x=st.session_state.timestamps,
            y=st.session_state.metric_history,
            mode='lines',
            name="Motor Current",
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Anomaly markers
    anomaly_times = [t for t, a in zip(st.session_state.timestamps, st.session_state.anomaly_flags) if a == 1]
    anomaly_values = [st.session_state.metric_history[i] for i, a in enumerate(st.session_state.anomaly_flags) if a == 1]
    
    if anomaly_times:
        fig.add_trace(
            go.Scatter(
                x=anomaly_times,
                y=anomaly_values,
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name="Anomalies"
            ),
            row=1, col=1
        )
    
    # Reconstruction loss plot
    if st.session_state.reconstruction_losses:
        fig.add_trace(
            go.Scatter(
                x=st.session_state.timestamps,
                y=st.session_state.reconstruction_losses,
                mode='lines+markers',
                name="Reconstruction Loss",
                line=dict(color='orange', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        margin=dict(t=50, l=10, r=10, b=10)
    )
    
    chart_placeholder.plotly_chart(fig, use_container_width=True)

# Main streaming logic
def process_next_point():
    if st.session_state.current_index >= len(df):
        st.session_state.is_running = False
        status_placeholder.info("📄 Reached end of dataset")
        return
    
    row = df.iloc[st.session_state.current_index]
    current_timestamp = datetime.now().strftime("%H:%M:%S")
    
    try:
        input_features = row[feature_columns].values.astype(float)
        st.session_state.buffer.append(input_features)
        st.session_state.timestamps.append(current_timestamp)
        st.session_state.metric_history.append(float(input_features[-1]))  # Motor_current
        
        if len(st.session_state.buffer) == SEQ_LEN:
            sequence = np.array(st.session_state.buffer)
            
            # Prepare API payload - match your FastAPI model
            payload = {
                "sequence": sequence.tolist(),  
                "timestamp": current_timestamp
            }
            
            try:
                response = requests.post(
                    API_ENDPOINT, 
                    json=payload,
                    timeout=5.0  
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    is_anomaly = response_data.get("anomaly", False)
                    reconstruction_loss = response_data.get("reconstruction_loss", 0.0)
                    
                    st.session_state.anomaly_flags.append(1 if is_anomaly else 0)
                    st.session_state.reconstruction_losses.append(reconstruction_loss)
                    
                    if is_anomaly:
                        # Display anomaly details
                        top_features = response_data.get("top_features", [])
                        anomaly_info = f"🚨 **Anomaly Detected at {current_timestamp}**\n"
                        anomaly_info += f"Loss: {reconstruction_loss:.6f}\n"
                        if top_features:
                            anomaly_info += "Top Contributing Features:\n"
                            for feat_name, feat_error in top_features:
                                anomaly_info += f"- {feat_name}: {feat_error:.6f}\n"
                        
                        status_placeholder.error(anomaly_info)
                    else:
                        status_placeholder.success(f"✅ Normal at {current_timestamp} (Loss: {reconstruction_loss:.6f})")
                        
                else:
                    st.session_state.anomaly_flags.append(0)
                    st.session_state.reconstruction_losses.append(0.0)
                    status_placeholder.warning(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.session_state.anomaly_flags.append(0)
                st.session_state.reconstruction_losses.append(0.0)
                status_placeholder.error(f"🔌 Connection Error: {str(e)}")
                
        else:
            # Not enough data yet
            st.session_state.anomaly_flags.append(0)
            st.session_state.reconstruction_losses.append(0.0)
            status_placeholder.info(f"📊 Collecting data... ({len(st.session_state.buffer)}/{SEQ_LEN})")
            
    except Exception as e:
        status_placeholder.error(f"💥 Processing Error: {str(e)}")
        st.session_state.anomaly_flags.append(0)
        st.session_state.reconstruction_losses.append(0.0)
    
    finally:
        st.session_state.current_index += 1

# Auto-refresh logic
if st.session_state.is_running:
    process_next_point()
    update_charts()
    time.sleep(INTERVAL)
    st.rerun()
else:
    # Update charts even when not running
    update_charts()

# Display recent anomalies
if st.session_state.anomaly_flags and any(st.session_state.anomaly_flags):
    st.subheader("🚨 Recent Anomalies")
    anomaly_data = []
    for i, (timestamp, flag, loss) in enumerate(zip(
        st.session_state.timestamps[-10:], 
        st.session_state.anomaly_flags[-10:], 
        st.session_state.reconstruction_losses[-10:]
    )):
        if flag:
            anomaly_data.append({
                "Timestamp": timestamp,
                "Reconstruction Loss": f"{loss:.6f}",
                "Motor Current": f"{st.session_state.metric_history[-(10-i)]:.2f}"
            })
    
    if anomaly_data:
        st.dataframe(pd.DataFrame(anomaly_data), use_container_width=True)

# API Status check
with st.sidebar:
    st.subheader("API Status")
    try:
        health_response = requests.get(f"{API_ENDPOINT.replace('/anomaly_check', '/health')}", timeout=2)
        if health_response.status_code == 200:
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Error")
    except:
        st.error("🔴 API Offline")