import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
import pandas as pd


class Engine(BaseModel):
    num_twf: int = Field(0, ge=0, le=1)
    num_hdf: int = Field(0, ge=0, le=1)
    num_pwf: int = Field(0, ge=0, le=1)
    num_osf: int = Field(0, ge=0, le=1)
    num_rnf: int = Field(0, ge=0, le=1)


class PredictResponse(BaseModel):
    condition_probability: float
    condition: bool

app = FastAPI(title="machine-failure-prediction")

# ✅ CORS must be added immediately after app initialization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] if hosting locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in, encoding='latin1')


def predict_single(machine_dict):
    default_values = {
        'num_twf': 0,
        'num_hdf': 0.0,
        'num_pwf': 0.0,
        'num_osf': 0.0,
        'num_rnf': 0.0
    }
    
    # Update defaults with provided values
    default_values.update(machine_dict)
    df = pd.DataFrame([default_values])
    result = pipeline.predict_proba(df)[0, 1]
    return float(result)


@app.post("/predict")
def predict(machine: Machine) -> PredictResponse:
    prob = predict_single(machine.model_dump())

    return PredictResponse(
        condition_probability=prob,
        condition=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)