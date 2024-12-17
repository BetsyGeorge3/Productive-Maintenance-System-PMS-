import Adafruit_DHT
import time

# Sensor Configuration
SENSOR_TYPE = Adafruit_DHT.DHT22  # Use DHT11 for DHT11 sensors
GPIO_PIN = 4  # GPIO pin connected to the DATA pin of the sensor

# Thresholds for Alerts (adjust based on your use case)
HUMIDITY_HIGH_THRESHOLD = 70.0  # High humidity threshold (%)
HUMIDITY_LOW_THRESHOLD = 30.0   # Low humidity threshold (%)

def read_humidity():
    """
    Reads humidity and temperature from the DHT sensor.
    Returns the values or None if there is an error.
    """
    humidity, temperature = Adafruit_DHT.read_retry(SENSOR_TYPE, GPIO_PIN)
    if humidity is not None and temperature is not None:
        return humidity, temperature
    else:
        print("Failed to read from the sensor. Retrying...")
        return None, None

def monitor_humidity():
    """
    Continuously monitor humidity and raise alerts if thresholds are breached.
    """
    print("Starting Humidity Monitoring...")
    while True:
        humidity, temperature = read_humidity()
        if humidity is not None:
            print(f"Humidity: {humidity:.2f}%, Temperature: {temperature:.2f}°C")
            
            # Check for high or low humidity
            if humidity > HUMIDITY_HIGH_THRESHOLD:
                print("Alert: High Humidity Detected!")
            elif humidity < HUMIDITY_LOW_THRESHOLD:
                print("Alert: Low Humidity Detected!")
        else:
            print("Sensor error. No data available.")
        
        time.sleep(2)  # Wait for 2 seconds before reading again

# Start monitoring
monitor_humidity()

