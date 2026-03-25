import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# 1. Create dummy data (In reality, this would be a SQL query or S3 download)
data = {
    'sqft': [1000, 1500, 2000, 2500, 3000, 3500],
    'bedrooms': [2, 3, 3, 4, 4, 5],
    'price': [300000, 450000, 500000, 650000, 700000, 850000]
}
df = pd.DataFrame(data)

# 2. Split into Features (X) and Target (y)
X = df[['sqft', 'bedrooms']]
y = df['price']

# 3. Train the Model
model = LinearRegression()
model.fit(X, y)

# 4. Save the artifact to a 'models' directory
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/house_model.joblib')

print("Model trained and saved to models/house_model.joblib")