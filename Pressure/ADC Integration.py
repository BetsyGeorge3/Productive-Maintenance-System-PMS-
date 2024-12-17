import spidev

# Initialize SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    """
    Read data from the ADC (MCP3008).
    :param channel: ADC channel (0-7)
    :return: ADC value (0-1023)
    """
    assert 0 <= channel <= 7, "ADC channel must be between 0 and 7"
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    value = ((adc[1] & 3) << 8) + adc[2]
    return value

def read_pressure_sensor():
    """
    Read pressure data from the sensor connected to ADC.
    Convert ADC value to pressure in kPa (based on sensor calibration).
    """
    adc_value = read_adc(0)  # Assuming sensor is connected to channel 0
    voltage = adc_value * (3.3 / 1023)  # Convert ADC value to voltage (3.3V range)
    pressure = (voltage / 3.3) * 700  # Convert voltage to pressure (MPX5700 range: 0-700 kPa)
    return pressure

