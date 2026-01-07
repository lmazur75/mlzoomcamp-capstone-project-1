import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd


class Machine(BaseModel):
    twf: int = Field(0, ge=0, le=1)
    hdf: int = Field(0, ge=0, le=1)
    pwf: int = Field(0, ge=0, le=1)
    osf: int = Field(0, ge=0, le=1)
    rnf: int = Field(0, ge=0, le=1)
    # Physical Columns (Defaults to "neutral" values from your notebook)
    # Adding these here ensures the API doesn't crash if they are missing
    air_temperature_k: float = Field(300.0, alias='air_temperature_[k]')
    process_temperature_k: float = Field(310.0, alias='process_temperature_[k]')
    rotational_speed_rpm: float = Field(1538.0, alias='rotational_speed_[rpm]')
    torque_nm: float = Field(40.0, alias='torque_[nm]')
    tool_wear_min: int = Field(108, alias='tool_wear_[min]')
    type: str = Field('L')
    
    class Config:
        populate_by_name = True

class PredictResponse(BaseModel):
    condition_probability: float
    condition: bool


app = FastAPI(title="machine-failure-prediction")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    pipeline = joblib.load('model.bin')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


def predict_single(machine_dict):
    # The model expects these EXACT column names from training
    # Looking at your notebook, after preprocessing, the columns are just the failure types
    """ default_values = {
        'twf': 0,
        'hdf': 0,
        'pwf': 0,
        'osf': 0,
        'rnf': 0,
        'air_temperature_[k]': 300.0,
        'process_temperature_[k]': 310.0,
        'rotational_speed_[rpm]': 1538.0,
        'torque_[nm]': 40.0,
        'tool_wear_[min]': 108.0,
        'type': 'L'
    } """
    

    
    # df = pd.DataFrame([default_values])
    df = pd.DataFrame([machine_dict])
    
    print(f"Input DataFrame:\n{df}")  # Debug print
    
    result = pipeline.predict_proba(df)[0, 1]
    return float(result)


@app.post("/predict")
def predict(machine: Machine) -> PredictResponse:
    data_for_model = machine.model_dump(by_alias=True)
    prob = predict_single(data_for_model)

    return PredictResponse(
        condition_probability=prob,
        condition=prob >= 0.5
    )


@app.get("/")
def read_root():
    return {"message": "Machine Failure Prediction API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)