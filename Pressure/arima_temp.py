import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Step 1: Load Data
data = pd.read_csv("temperature_data.csv", parse_dates=["timestamp"], index_col="timestamp")
data = data.sort_index()
temperature_series = data["temperature"]

# Plot data
plt.figure(figsize=(12, 6))
plt.plot(temperature_series, label="Temperature")
plt.title("Temperature Time Series")
plt.xlabel("Timestamp")
plt.ylabel("Temperature (°F)")
plt.legend()
plt.show()

# Step 2: Check Stationarity
def check_stationarity(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary. Differencing may be required.")

check_stationarity(temperature_series)

# Perform differencing if necessary
if not adfuller(temperature_series)[1] <= 0.05:
    diff_series = temperature_series.diff().dropna()
    check_stationarity(diff_series)

# Step 3: Find Optimal ARIMA Parameters
auto_model = auto_arima(temperature_series, seasonal=False, stepwise=True, trace=True)
print("Optimal ARIMA Parameters:", auto_model.order)

# Step 4: Train ARIMA Model
p, d, q = auto_model.order
model = ARIMA(temperature_series, order=(p, d, q))
fitted_model = model.fit()

# Print model summary
print(fitted_model.summary())

# Step 5: Forecast Future Temperatures
forecast_steps = 30  # Forecast for the next 30 days
forecast = fitted_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=temperature_series.index[-1], periods=forecast_steps + 1, freq="D")[1:]
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(temperature_series, label="Historical Data")
plt.plot(forecast_index, forecast_values, label="Forecast", color="orange")
plt.fill_between(forecast_index, 
                 forecast_conf_int.iloc[:, 0], 
                 forecast_conf_int.iloc[:, 1], 
                 color="orange", alpha=0.2, label="Confidence Interval")
plt.title("Temperature Forecast Using ARIMA")
plt.xlabel("Timestamp")
plt.ylabel("Temperature (°F)")
plt.legend()
plt.show()

