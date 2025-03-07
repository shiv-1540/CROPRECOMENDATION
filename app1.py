from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load the trained RandomForest model
with open("RF.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize FastAPI
app = FastAPI(title="Smart Crop Recommendation API")

# Define the input data format using Pydantic
class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain when deploying (e.g., ["https://yourfrontend.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction endpoint
@app.post("/predict")
def predict_crop(data: CropInput):
    input_data = np.array([[data.nitrogen, data.phosphorus, data.potassium,
                            data.temperature, data.humidity, data.ph, data.rainfall]])
    
    prediction = model.predict(input_data)
    return {"recommended_crop": prediction[0]}

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Smart Crop Recommendation API"}
