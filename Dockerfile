# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and the prediction script
# Note: You must run 'python train.py' locally first to create the models folder
COPY models/house_model.joblib ./models/
COPY predict.py .

# Command to run the prediction
CMD ["python", "predict.py"]