import pandas as pd

# Load preprocessed data
data = pd.read_csv("preprocessed_temperature_data.csv")

# Create new features
data["temperature_trend"] = data["temperature_smoothed"].diff()  # Rate of change
data["temperature_deviation"] = data["temperature_smoothed"] - data["temperature_smoothed"].mean()  # Deviation

# Drop NaN values created by differencing
data.dropna(inplace=True)

# Save the engineered dataset
data.to_csv("temperature_data_with_features.csv", index=False)
print("Feature-engineered data saved to 'temperature_data_with_features.csv'.")

