import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load humidity dataset (replace with your dataset)
# Example: The dataset should have columns like "timestamp", "humidity"
data = pd.read_csv("humidity_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.set_index("timestamp")

# Visualize the data
plt.plot(data["humidity"])
plt.title("Humidity Levels Over Time")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.show()

# Scale the data to [0, 1] using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences for time-series modeling
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Define the sequence length (e.g., 24 hours if data is hourly)
sequence_length = 24
X, y = create_sequences(data_scaled, sequence_length)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)  # Single output for humidity prediction
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Predict on test data
predicted = model.predict(X_test)

# Reverse scaling to get original humidity values
predicted_humidity = scaler.inverse_transform(predicted)
actual_humidity = scaler.inverse_transform(y_test)

# Visualize predictions
plt.plot(actual_humidity, label="Actual Humidity")
plt.plot(predicted_humidity, label="Predicted Humidity")
plt.title("Actual vs Predicted Humidity")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.legend()
plt.show()

# Generate future predictions
def generate_future_data(model, data, steps, scaler):
    future_data = data[-sequence_length:]
    predictions = []
    for _ in range(steps):
        prediction = model.predict(future_data[np.newaxis, :, :])[0]
        predictions.append(prediction)
        future_data = np.append(future_data[1:], prediction, axis=0)
    return scaler.inverse_transform(predictions)

# Generate 24 future steps (e.g., next 24 hours of humidity)
future_humidity = generate_future_data(model, data_scaled, steps=24, scaler=scaler)

# Visualize future predictions
plt.plot(future_humidity, label="Future Humidity")
plt.title("Generated Future Humidity")
plt.xlabel("Time Steps")
plt.ylabel("Humidity (%)")
plt.legend()
plt.show()

model.save("humidity_lstm_model.h5")

