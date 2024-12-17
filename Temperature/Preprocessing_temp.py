import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data
data = pd.read_csv("temperature_data.csv")
scaler = MinMaxScaler()

# Normalize temperature
data["temperature_normalized"] = scaler.fit_transform(data[["temperature"]])

# Smooth noise using a rolling average
data["temperature_smoothed"] = data["temperature_normalized"].rolling(window=5).mean()
data.dropna(inplace=True)

# Save preprocessed data
data.to_csv("preprocessed_temperature_data.csv", index=False)
print("Preprocessed temperature data saved to 'preprocessed_temperature_data.csv'.")

