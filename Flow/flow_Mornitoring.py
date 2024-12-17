import time
import numpy as np
import matplotlib.pyplot as plt

# Simulate or Read Data from Flow Sensor
def read_flow_data_simulated(duration=60, interval=1):
    """
    Simulate flow sensor data for a given duration and interval.
    Real implementation should read from the sensor instead.
    """
    timestamps = []
    flow_rates = []
    start_time = time.time()

    while time.time() - start_time < duration:
        # Simulated flow rate in liters per minute (random + sinusoidal variation)
        simulated_flow = 10 + 2 * np.sin(time.time()) + np.random.normal(0, 0.5)
        timestamps.append(time.time() - start_time)
        flow_rates.append(simulated_flow)
        
        print(f"Time: {timestamps[-1]:.2f}s, Flow Rate: {simulated_flow:.2f} L/min")
        time.sleep(interval)

    return np.array(timestamps), np.array(flow_rates)

def preprocess_flow_data(flow_rates):
    """
    Normalize and filter noise from flow data.
    """
    # Normalize data
    normalized_data = (flow_rates - np.min(flow_rates)) / (np.max(flow_rates) - np.min(flow_rates))

    # Simple moving average for noise reduction
    smoothed_data = np.convolve(normalized_data, np.ones(5) / 5, mode='same')

    return smoothed_data


def detect_anomalies(flow_rates, threshold=0.2):
    """
    Detect anomalies in flow rates based on deviation from the mean.
    """
    mean_flow = np.mean(flow_rates)
    deviation = np.abs(flow_rates - mean_flow)
    anomalies = deviation > threshold * mean_flow

    return anomalies


def plot_flow_data(timestamps, original_data, smoothed_data, anomalies):
    """
    Plot the original, smoothed flow data, and highlight anomalies.
    """
    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.plot(timestamps, original_data, label="Original Flow Data", alpha=0.5)

    # Plot smoothed data
    plt.plot(timestamps, smoothed_data, label="Smoothed Flow Data", linewidth=2)

    # Highlight anomalies
    anomaly_times = timestamps[anomalies]
    anomaly_values = smoothed_data[anomalies]
    plt.scatter(anomaly_times, anomaly_values, color="red", label="Anomalies", zorder=5)

    plt.xlabel("Time (s)")
    plt.ylabel("Flow Rate (L/min)")
    plt.title("Flow Rate Monitoring with Anomalies")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Simulate flow data for 60 seconds
    print("Simulating flow sensor data...")
    timestamps, flow_data = read_flow_data_simulated(duration=60, interval=1)

    # Preprocess the flow data
    smoothed_flow_data = preprocess_flow_data(flow_data)

    # Detect anomalies
    anomalies = detect_anomalies(smoothed_flow_data, threshold=0.2)

    # Plot results
    print("Plotting flow data and anomalies...")
    plot_flow_data(timestamps, flow_data, smoothed_flow_data, anomalies)

