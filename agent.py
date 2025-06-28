
from langgraph.graph import StateGraph, END, START
from typing import TypedDict
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import requests

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

autoencoder = load_model("autoencoder_model.keras")
scaler = joblib.load("scaler.pkl")
threshold = 0.3569

feature_columns = [
    'TP2', 'TP3', 'H1', 'DV_pressure',
    'Reservoirs', 'Oil_temperature', 'Motor_current'
]

class AgentState(TypedDict):
    sequence: np.ndarray
    prediction: np.ndarray
    reconstruction_loss: float
    top_features: list[tuple[str, float]]
    timestamp: str
    anomaly: bool
    
def run_inference(state: AgentState) -> AgentState:
    X_seq = state['sequence']
    pred = autoencoder.predict(np.array([X_seq]), verbose=0)  
    loss = np.mean((X_seq - pred[0]) ** 2)
    state['reconstruction_loss'] = loss
    state['prediction'] = pred[0]
    return state

def check_anomaly(state: AgentState) -> str:  # Added return type annotation
    return "explain" if state['reconstruction_loss'] > threshold else END

def explain_features(state: AgentState) -> AgentState:
    seq = state['sequence']
    pred = state['prediction']
    errors = np.mean((seq - pred) ** 2, axis=0)
    top_feats = sorted(zip(feature_columns, errors), key=lambda x: -x[1])[:3]
    state['top_features'] = top_feats
    return state

def notify_agent(state: AgentState) -> AgentState:
    top_feats = state['top_features']
    loss = state['reconstruction_loss']
    timestamp = state.get("timestamp", 'N/A')
    state['anomaly'] = True
    message = f"\n🚨 Anomaly Detected at {timestamp}!\nLoss: {loss:.6f}\nTop Contributing Features:\n"
    for feat, err in top_feats:
        message += f"- {feat}: {err:.6f}\n"
    send_alerts(message)
    return state

def convert_state_to_json_serializable(state: AgentState) -> dict:
    json_state = {}
    
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            json_state[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            json_state[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            json_state[key] = int(value)
        elif isinstance(value, np.bool_):
            json_state[key] = bool(value)
        elif key == 'top_features' and isinstance(value, list):
            # Convert feature tuples to JSON-serializable format
            json_state[key] = [(feat, float(err)) for feat, err in value]
        else:
            json_state[key] = value
    
    return json_state

def send_alerts(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.status_code == 200

graph = StateGraph(AgentState)
graph.add_node("inference", run_inference)
graph.add_node("explain", explain_features)
graph.add_node("notify", notify_agent)

graph.set_entry_point("inference")

graph.add_conditional_edges(
    source="inference",
    path=check_anomaly,
    path_map={
        "explain": "explain",
        END: END
    }
)
graph.add_edge("explain", "notify")
graph.add_edge("notify", END)

langgraph_app = graph.compile()
