import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import os
import logging

# Set up logging to track progress in CI/CD logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_model(data_path='data/latest_market_data.csv', model_save_path='models/house_model.joblib'):
    # 1. Load Data
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at {data_path}. Training aborted.")
        return

    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Split Features and Target
    X = df[['sqft', 'bedrooms']]
    y = df['price']

    # 3. Train/Test Split (Important for validating if the model actually works)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train the Model
    logging.info("Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Model Validation (The "Quality Gate")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"Model Training Complete. Mean Absolute Error: {mae}")

    # 6. Save the Artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    logging.info(f"Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    train_model()