import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import os
import logging
import datetime

# Set up logging for CI/CD visibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_model(data_path='data/latest_market_data.csv', model_save_path='models/house_model.joblib'):
    # 1. Load Data
    if not os.path.exists(data_path):
        logging.error(f"❌ Data file not found at {data_path}. Training aborted.")
        return

    logging.info(f"📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Split Features (X) and Target (y)
    # We are predicting 'price' based on 'sqft' and 'bedrooms'
    X = df[['sqft', 'bedrooms']]
    y = df['price']

    # 3. Train/Test Split
    # We hold back 20% of data to 'test' the model's honesty
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train the Model
    logging.info("🧠 Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Model Validation (The Quality Gate)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"📊 Model Training Complete. Mean Absolute Error (MAE): {mae}")

    # 6. Save the Error Score for GitHub Actions to read
    with open("model_score.txt", "w") as f:
        f.write(str(mae))
    logging.info("📝 Error score saved to model_score.txt")

    # 7. Save the Model Artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    logging.info(f"✅ Model artifact saved to {model_save_path}")
    # 8. Save the Model Artifact with a timestamp (Versioned)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = f'models/house_model_{timestamp}.joblib'
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, versioned_path)
    # Also save a 'latest' copy for the API to use
    joblib.dump(model, 'models/house_model_latest.joblib')
    
    logging.info(f"✅ Versioned model saved: {versioned_path}")

if __name__ == "__main__":
    train_model()