from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
import uvicorn

with open("/app/model.pkl", "rb") as f:
    model = pickle.load(f)


data = pd.read_csv('/app/data/processed/transformed/xtrainT.csv')


# Definición del modelo de entrada
class modelData(BaseModel):
    features: List[float]

# Crear la aplicación FastAPI
app = FastAPI()

@app.post("/predict")
def predict(ml_data: modelData):
    if len(ml_data.features) != model.n_features_in_:
        raise HTTPException(status_code=400,
                            detail=f"Input must contain {model.n_features_in_} features."
                            )
    prediction = model.predict([ml_data.features])[0]
    # Aquí corregimos el acceso a target_names usando iloc

    return {"prediction": int(prediction)}

@app.get("/")
def read_root():
    return {"message": "MLOps equipo 14 Model API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)