import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load flow data (simulate data or load from a file)
def load_flow_data():
    # Simulating some flow data for this example
    time_steps = 1000
    flow_rate = 10 + 2 * np.sin(np.linspace(0, 50, time_steps)) + np.random.normal(0, 0.5, time_steps)
    
    return flow_rate

# Preprocess and normalize data
def preprocess_flow_data(flow_data):
    # Reshaping the data to match the input for the neural network
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Normalize to [-1, 1]
    flow_data_scaled = scaler.fit_transform(flow_data.reshape(-1, 1))
    return flow_data_scaled

# Load and preprocess the flow data
flow_data = load_flow_data()
flow_data_scaled = preprocess_flow_data(flow_data)

# Visualizing the flow data
plt.plot(flow_data)
plt.title("Original Flow Data")
plt.xlabel("Time")
plt.ylabel("Flow Rate (L/min)")
plt.show()


