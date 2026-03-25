import joblib

# Load the saved model
model = joblib.load('models/house_model.joblib')

# Predict for a 2000 sqft, 3 bedroom house
prediction = model.predict([[2000, 3]])
print(f"Predicted Price: ${prediction[0]:,.2f}")