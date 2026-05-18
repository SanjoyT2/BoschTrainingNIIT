from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os

class DefectFeatures(BaseModel):
    product_id: int
    defect_type: str
    defect_location: str
    inspection_method: str
    month: int
    weekday: str
    is_weekend: int
    product_defect_count: int

app = FastAPI(title="Defects ML API")

MODEL_SEVERITY_PATH = 'severity_model.pkl'
MODEL_COST_PATH = 'cost_model.pkl'


def load_model(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

severity_model = load_model(MODEL_SEVERITY_PATH)
cost_model = load_model(MODEL_COST_PATH)


@app.on_event("startup")
def check_models():
    if severity_model is None:
        print(f"Warning: {MODEL_SEVERITY_PATH} not found. Run the notebook to create it.")
    if cost_model is None:
        print(f"Warning: {MODEL_COST_PATH} not found. Run the notebook to create it.")


@app.get("/health")
def health():
    return {
        "severity_model": severity_model is not None,
        "cost_model": cost_model is not None
    }


@app.post("/predict/severity")
def predict_severity(features: DefectFeatures):
    if severity_model is None:
        raise HTTPException(status_code=503, detail="Severity model not available")
    df = pd.DataFrame([features.dict()])
    pred = severity_model.predict(df)
    return {"prediction": int(pred[0])}


@app.post("/predict/cost")
def predict_cost(features: DefectFeatures):
    if cost_model is None:
        raise HTTPException(status_code=503, detail="Cost model not available")
    df = pd.DataFrame([features.dict()])
    pred = cost_model.predict(df)
    return {"prediction": float(pred[0])}
