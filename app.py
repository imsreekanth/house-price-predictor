from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the model once when the server starts
model = joblib.load('models/house_model.joblib')

@app.get("/")
def home():
    return {"message": "House Price Prediction API is Running"}

@app.post("/predict")
def predict(data: dict):
    # Data expected: {"sqft": 2000, "bedrooms": 3}
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}