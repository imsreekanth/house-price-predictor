import pandas as pd
import os

# Imagine this data reflects NEW market prices (Prices have doubled!)
new_market_data = {
    'sqft': [1000, 1500, 2000, 2500, 3000, 3500],
    'bedrooms': [2, 3, 3, 4, 4, 5],
    'price': [600000, 900000, 1000000, 1300000, 1400000, 1700000] # Prices Doubled!
}

df = pd.DataFrame(new_market_data)
df.to_csv('data/latest_market_data.csv', index=False)
print("New data arrived in 'data/' folder. Time to retrain!")