from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the "Latest" model
model = joblib.load('models/house_model_latest.joblib')

@app.get("/health")
def health():
    # In MLOps, a health check can also verify model version
    return {"status": "healthy", "model_version": "latest"}

@app.post("/predict")
def predict(data: dict):
    # Log the incoming request (This is "Inference Logging")
    print(f"Request received: {data}") 
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}