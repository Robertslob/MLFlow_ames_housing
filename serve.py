from fastapi import FastAPI
import mlflow
import pandas as pd
import numpy as np
from config import MODEL_NAME
from src.features import prepare_X

app = FastAPI()

model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@champion")

@app.post("/predict")
async def predict(payload: dict):
    X = pd.DataFrame([payload])
    X, _ = prepare_X(X)
        
    pred_log = model.predict(X, num_iteration=model.best_iteration_)
    pred = np.expm1(pred_log)
    return {"prediction": float(pred[0])}

@app.get("/")
async def root():
    return {"message": "API to predict Ames Housing prices using the current champion LightGBM model."}

@app.post("/reload-model")
def reload_model():
    global model
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@champion")
    return {"status": "reloaded"}