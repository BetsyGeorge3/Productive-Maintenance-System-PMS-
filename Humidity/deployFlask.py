from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json

# Load the trained LSTM model
model = tf.keras.models.load_model("humidity_lstm_model.h5")

# Create a FastAPI app
app = FastAPI()

# Define thresholds for alerts
HIGH_HUMIDITY_THRESHOLD = 80  # Example: 80% humidity
LOW_HUMIDITY_THRESHOLD = 20  # Example: 20% humidity

# Define the input format for real-time data
class HumidityData(BaseModel):
    readings: list  # List of recent humidity readings (last 24 values)
    timestamp: str  # Timestamp of the most recent reading

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Humidity Monitoring API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict_humidity(data: HumidityData):
    # Validate input data length
    if len(data.readings) < 24:
        raise HTTPException(status_code=400, detail="At least 24 readings are required.")

    # Preprocess input data (normalize between 0 and 1)
    readings_array = np.array(data.readings).reshape(-1, 1)
    scaler = tf.keras.models.load_model("scaler.pkl")  # Load the scaler if saved earlier
    readings_scaled = scaler.transform(readings_array)

    # Prepare input sequence for prediction
    input_sequence = readings_scaled[-24:]  # Take the last 24 readings
    input_sequence = input_sequence.reshape(1, 24, 1)

    # Make the prediction
    predicted_value_scaled = model.predict(input_sequence)[0]
    predicted_value = scaler.inverse_transform(predicted_value_scaled.reshape(-1, 1))[0][0]

    # Trigger alerts based on thresholds
    alert = None
    if predicted_value > HIGH_HUMIDITY_THRESHOLD:
        alert = f"High Humidity Alert! Predicted: {predicted_value:.2f}%"
    elif predicted_value < LOW_HUMIDITY_THRESHOLD:
        alert = f"Low Humidity Alert! Predicted: {predicted_value:.2f}%"

    return {
        "timestamp": data.timestamp,
        "predicted_humidity": round(predicted_value, 2),
        "alert": alert,
    }

