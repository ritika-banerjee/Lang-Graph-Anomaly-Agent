from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import numpy as np
from agent import langgraph_app, convert_state_to_json_serializable
from typing import List

app = FastAPI()

class SequencePayload(BaseModel):
    sequence: List[List[float]]
    timestamp: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
@app.get("/health")
def get_health():
    return {"status": "working"}
    
@app.post("/anomaly_check")
def anomaly_check(payload: SequencePayload):
    input_sequence = np.array(payload.sequence)
    state = {
        "sequence": input_sequence,
        "timestamp": payload.timestamp,
        "prediction": None,
        "reconstruction_loss": 0.0,
        "top_features": [],
        "anomaly": False
    }

    result = langgraph_app.invoke(state)
    json_result = convert_state_to_json_serializable(result)
    
    return json_result
    