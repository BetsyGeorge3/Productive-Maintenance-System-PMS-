import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Simulate Pressure Data
np.random.seed(42)
time_points = 500
pressure_data = 100 + np.random.normal(loc=0, scale=5, size=time_points)  # Normal pressure with noise

# Add anomalies
anomalies = np.random.choice(time_points, size=10, replace=False)
pressure_data[anomalies] += np.random.normal(loc=20, scale=5, size=10)

# Create DataFrame
pressure_df = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=time_points, freq='H'),
    'pressure': pressure_data
})

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(pressure_df['timestamp'], pressure_df['pressure'], label='Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure (kPa)')
plt.title('Pressure Data with Anomalies')
plt.legend()
plt.grid()
plt.show()

# Train-test split
train_size = int(len(pressure_df) * 0.8)
train_data = pressure_df['pressure'][:train_size]
test_data = pressure_df['pressure'][train_size:]

# Fit ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))  # ARIMA(p, d, q): Adjust p, d, q based on your data
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast
forecast_steps = len(test_data)
forecast = model_fit.forecast(steps=forecast_steps)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(pressure_df['timestamp'], pressure_df['pressure'], label='Actual')
plt.plot(pressure_df['timestamp'][train_size:], forecast, color='red', label='Forecast')
plt.xlabel('Time')
plt.ylabel('Pressure (kPa)')
plt.title('ARIMA Model - Pressure Forecasting')
plt.legend()
plt.grid()
plt.show()

# Evaluate model
mse = mean_squared_error(test_data, forecast)
print(f"Mean Squared Error: {mse:.2f}")


# Calculate residuals (errors)
residuals = test_data.values - forecast
threshold = 3 * np.std(residuals)  # Define anomaly threshold (e.g., 3 standard deviations)

# Identify anomalies
anomalies = pressure_df.iloc[train_size:][abs(residuals) > threshold]

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.plot(pressure_df['timestamp'], pressure_df['pressure'], label='Pressure')
plt.scatter(anomalies['timestamp'], anomalies['pressure'], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Pressure (kPa)')
plt.title('Anomaly Detection with ARIMA')
plt.legend()
plt.grid()
plt.show()


