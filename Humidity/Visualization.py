import matplotlib.pyplot as plt
from collections import deque

# Store data for plotting
humidity_data = deque(maxlen=50)
time_data = deque(maxlen=50)

def plot_humidity():
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, humidity_data, label="Humidity (%)", color='blue')
    plt.axhline(HUMIDITY_HIGH_THRESHOLD, color='red', linestyle='--', label="High Threshold")
    plt.axhline(HUMIDITY_LOW_THRESHOLD, color='green', linestyle='--', label="Low Threshold")
    plt.title("Real-Time Humidity Monitoring")
    plt.xlabel("Time")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.grid()
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()

# Modify monitor_humidity to include plotting
def monitor_humidity_with_plot():
    print("Starting Humidity Monitoring with Visualization...")
    while True:
        humidity, temperature = read_humidity()
        if humidity is not None:
            print(f"Humidity: {humidity:.2f}%, Temperature: {temperature:.2f}°C")
            humidity_data.append(humidity)
            time_data.append(time.time())
            plot_humidity()
            
            # Check for high or low humidity
            if humidity > HUMIDITY_HIGH_THRESHOLD:
                print("Alert: High Humidity Detected!")
            elif humidity < HUMIDITY_LOW_THRESHOLD:
                print("Alert: Low Humidity Detected!")
        else:
            print("Sensor error. No data available.")
        
        time.sleep(2)  # Wait for 2 seconds before reading again

# Start monitoring with plotting
monitor_humidity_with_plot()

