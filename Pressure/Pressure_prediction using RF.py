import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Simulate pressure data
def simulate_pressure_data(samples=1000):
    """
    Simulate pressure data with normal and anomaly points.
    """
    np.random.seed(42)
    normal_pressure = np.random.normal(loc=100, scale=5, size=int(samples * 0.9))  # Normal data
    anomaly_pressure = np.random.normal(loc=150, scale=10, size=int(samples * 0.1))  # Anomaly data
    pressure = np.concatenate([normal_pressure, anomaly_pressure])
    labels = np.concatenate([np.zeros(len(normal_pressure)), np.ones(len(anomaly_pressure))])  # 0: Normal, 1: Anomaly
    timestamps = np.arange(samples)  # Simulated time
    return pd.DataFrame({'timestamp': timestamps, 'pressure': pressure, 'label': labels})

# Generate data
data = simulate_pressure_data(1000)

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['pressure'], label='Pressure', alpha=0.7)
plt.scatter(data['timestamp'][data['label'] == 1], data['pressure'][data['label'] == 1], color='red', label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Pressure (kPa)')
plt.title('Simulated Pressure Data with Anomalies')
plt.legend()
plt.grid()
plt.show()


# Feature Engineering
def add_features(df):
    """
    Add derived features to the dataset.
    """
    df['pressure_mean_10'] = df['pressure'].rolling(window=10).mean()  # Moving average (10 samples)
    df['pressure_std_10'] = df['pressure'].rolling(window=10).std()  # Moving standard deviation
    df['pressure_diff'] = df['pressure'].diff()  # Pressure difference between consecutive readings
    df['pressure_dev_from_nominal'] = abs(df['pressure'] - 100)  # Deviation from nominal pressure
    df = df.fillna(0)  # Handle NaN values from rolling operations
    return df

# Add features to data
data = add_features(data)


# Prepare data for training
features = ['pressure', 'pressure_mean_10', 'pressure_std_10', 'pressure_diff', 'pressure_dev_from_nominal']
X = data[features]
y = data['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Simulate new pressure data
new_data = simulate_pressure_data(200)
new_data = add_features(new_data)
X_new = scaler.transform(new_data[features])

# Predict anomalies
new_data['predicted_label'] = model.predict(X_new)

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(new_data['timestamp'], new_data['pressure'], label='Pressure', alpha=0.7)
plt.scatter(new_data['timestamp'][new_data['predicted_label'] == 1], new_data['pressure'][new_data['predicted_label'] == 1],
            color='red', label='Predicted Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Pressure (kPa)')
plt.title('Predicted Anomalies in New Pressure Data')
plt.legend()
plt.grid()
plt.show()


