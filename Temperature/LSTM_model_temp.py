import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load feature-engineered data
data = pd.read_csv("temperature_data_with_features.csv")

# Prepare data for LSTM
sequence_length = 10
X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data[["temperature_smoothed", "temperature_trend", "temperature_deviation"]].iloc[i:i + sequence_length].values)
    y.append(data["failure"].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(50),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save("lstm_temperature_prediction_model.h5")
print("LSTM model saved to 'lstm_temperature_prediction_model.h5'.")

