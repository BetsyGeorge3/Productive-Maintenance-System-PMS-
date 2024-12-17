import time
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Load ARIMA model (pre-trained)
model_fit = model.fit()  # Replace with pre-saved ARIMA model if available

# Simulate real-time data stream
def stream_pressure_data():
    """Simulates a real-time stream of pressure data."""
    while True:
        yield 100 + np.random.normal(0, 5)  # Replace with actual sensor data stream
        time.sleep(1)  # Simulate 1-second delay between readings

# Real-time analysis
threshold = 3 * np.std(residuals)  # Anomaly threshold based on historical data

print("Starting real-time pressure monitoring...")
for pressure in stream_pressure_data():
    # Forecast next value
    forecast = model_fit.forecast(steps=1)[0]
    residual = pressure - forecast

    # Detect anomalies
    if abs(residual) > threshold:
        print(f"Anomaly Detected! Pressure: {pressure:.2f}, Forecast: {forecast:.2f}, Residual: {residual:.2f}")
    else:
        print(f"Pressure: {pressure:.2f}, Forecast: {forecast:.2f}")

