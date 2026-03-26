import pandas as pd
import os

os.makedirs('data', exist_ok=True)

# These prices are 2x higher than what the model learned.
# This will cause the Mean Absolute Error (MAE) to skyrocket.
drifted_data = {
    'sqft': [1000, 1500, 2000, 2500, 3000, 3500],
    'bedrooms': [2, 3, 3, 4, 4, 5],
    'price': [1200000, 1800000, 2000000, 2600000, 2800000, 3400000] 
}

df = pd.DataFrame(drifted_data)
df.to_csv('data/latest_market_data.csv', index=False)
print("🚨 Drifted data injected into data/latest_market_data.csv")