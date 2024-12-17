import time
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

# Sensor parameters
VOLTAGE_REFERENCE = 5.0  # Reference voltage of ADC in volts
ADC_RESOLUTION = 1024.0  # ADC resolution (10-bit ADC)

# ACS712 Current sensor parameters
ACS712_SENSITIVITY = 185.0  # mV per ampere for ACS712-05B

# Current transformer parameters (CT100)
CT_RATIO = 1000  # Example: 1000:1 current ratio

# Thresholds for alerts
OVERLOAD_THRESHOLD = 10.0  # Current overload threshold in amperes
VOLTAGE_IMBALANCE_THRESHOLD = 220.0  # Voltage imbalance threshold in volts

# Mock function to read ADC data (replace with actual ADC reading logic)
def read_adc(sensor_pin):
    # Simulating ADC reading (replace with GPIO library code)
    return np.random.randint(480, 520)  # Example: Random values near midpoint

# Function to convert ADC reading to current (ACS712)
def adc_to_current(adc_value):
    voltage = (adc_value / ADC_RESOLUTION) * VOLTAGE_REFERENCE
    current = (voltage - (VOLTAGE_REFERENCE / 2)) / (ACS712_SENSITIVITY / 1000)
    return current

# Function to convert ADC reading to voltage (CT100)
def adc_to_voltage(adc_value):
    voltage = (adc_value / ADC_RESOLUTION) * VOLTAGE_REFERENCE
    actual_voltage = voltage * CT_RATIO
    return actual_voltage

# Monitoring function
def monitor_current_voltage():
    current_readings = []
    voltage_readings = []

    # Collect data for a specific duration (e.g., 10 seconds)
    start_time = time.time()
    while time.time() - start_time < 10:
        # Read data from sensors
        adc_current = read_adc(sensor_pin=0)  # Replace with actual sensor pin
        adc_voltage = read_adc(sensor_pin=1)  # Replace with actual sensor pin

        # Convert ADC values to current and voltage
        current = adc_to_current(adc_current)
        voltage = adc_to_voltage(adc_voltage)

        current_readings.append(current)
        voltage_readings.append(voltage)

        # Print real-time values
        print(f"Current: {current:.2f} A, Voltage: {voltage:.2f} V")

        # Check for anomalies
        if current > OVERLOAD_THRESHOLD:
            print("⚠️ Alert: Current overload detected!")
        if voltage < VOLTAGE_IMBALANCE_THRESHOLD:
            print("⚠️ Alert: Voltage imbalance detected!")

        time.sleep(0.5)

    # Compute average current and voltage
    avg_current = mean(current_readings)
    avg_voltage = mean(voltage_readings)
    print(f"\nAverage Current: {avg_current:.2f} A")
    print(f"Average Voltage: {avg_voltage:.2f} V")

    # Plot data for analysis
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(current_readings, label="Current (A)", color="blue")
    plt.axhline(OVERLOAD_THRESHOLD, color="red", linestyle="--", label="Overload Threshold")
    plt.legend()
    plt.title("Current Monitoring")
    plt.ylabel("Current (A)")

    plt.subplot(2, 1, 2)
    plt.plot(voltage_readings, label="Voltage (V)", color="green")
    plt.axhline(VOLTAGE_IMBALANCE_THRESHOLD, color="red", linestyle="--", label="Imbalance Threshold")
    plt.legend()
    plt.title("Voltage Monitoring")
    plt.xlabel("Time (samples)")
    plt.ylabel("Voltage (V)")

    plt.tight_layout()
    plt.show()

# Run the monitoring system
if __name__ == "__main__":
    print("Starting Current and Voltage Monitoring System...")
    monitor_current_voltage()

