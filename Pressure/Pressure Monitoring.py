import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Simulated Pressure Sensor Reading (Replace this with actual sensor data acquisition code)
def read_pressure_sensor():
    """
    Simulate pressure sensor readings. Replace this function with actual
    sensor data reading logic using ADC or GPIO library (e.g., spidev for Raspberry Pi).
    """
    # Simulated pressure data (in kPa) with some random noise
    true_pressure = 100  # Nominal pressure value in kPa
    noise = np.random.normal(0, 2, 1)[0]  # Gaussian noise
    return true_pressure + noise

# Anomaly Detection Function
def detect_anomaly(pressure, threshold=10):
    """
    Detect anomalies in pressure readings.
    :param pressure: Current pressure reading (kPa)
    :param threshold: Allowed deviation from nominal pressure (kPa)
    :return: True if anomaly is detected, False otherwise
    """
    nominal_pressure = 100  # Nominal operating pressure (kPa)
    deviation = abs(pressure - nominal_pressure)
    return deviation > threshold

# Data Collection and Monitoring Loop
pressure_readings = []
anomaly_flags = []
timestamps = []

print("Starting pressure monitoring... Press Ctrl+C to stop.")

try:
    while True:
        # Step 1: Read pressure sensor
        pressure = read_pressure_sensor()

        # Step 2: Detect anomalies
        anomaly = detect_anomaly(pressure)

        # Step 3: Log data
        current_time = time.time()
        pressure_readings.append(pressure)
        anomaly_flags.append(anomaly)
        timestamps.append(current_time)

        # Step 4: Display real-time status
        if anomaly:
            print(f"ALERT! Pressure anomaly detected: {pressure:.2f} kPa")
        else:
            print(f"Pressure: {pressure:.2f} kPa - Normal")

        # Wait for the next reading (1 second interval)
        time.sleep(1)

except KeyboardInterrupt:
    print("\nMonitoring stopped. Analyzing collected data...")

# Data Analysis and Visualization
pressure_readings = np.array(pressure_readings)
timestamps = np.array(timestamps)

# Apply Savitzky-Golay filter to smooth the pressure data
smoothed_pressure = savgol_filter(pressure_readings, window_length=11, polyorder=2)

# Plot the pressure data
plt.figure(figsize=(12, 6))
plt.plot(timestamps, pressure_readings, label='Raw Pressure Data', alpha=0.6)
plt.plot(timestamps, smoothed_pressure, label='Smoothed Pressure Data', color='red')
plt.axhline(y=110, color='orange', linestyle='--', label='Anomaly Threshold (High)')
plt.axhline(y=90, color='orange', linestyle='--', label='Anomaly Threshold (Low)')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (kPa)')
plt.title('Pressure Monitoring with Anomaly Detection')
plt.legend()
plt.grid()
plt.show()

