from fastapi import FastAPI, HTTPException
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the trained model
with open('RF.pkl', 'rb') as model_file:
    RF_Model_pkl = pickle.load(model_file)

# Define the input data model using Pydantic
class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict_crop(input_data: CropInput):
    try:
        # Convert input data to numpy array
        inputs = np.array([
            input_data.nitrogen,
            input_data.phosphorus,
            input_data.potassium,
            input_data.temperature,
            input_data.humidity,
            input_data.ph,
            input_data.rainfall
        ]).reshape(1, -1)

        # Make prediction
        prediction = RF_Model_pkl.predict(inputs)
        return {"recommended_crop": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/")
def shiv():
    print("Hello to ShivWorld..!")
# Run the API using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)