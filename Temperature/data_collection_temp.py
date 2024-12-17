import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic temperature data
np.random.seed(42)
num_days = 365
timestamps = [datetime.now() - timedelta(days=i) for i in range(num_days)]
temperatures = np.random.normal(loc=70, scale=5, size=num_days)  # Avg temp: 70°F, SD: 5°F
failures = np.random.choice([0, 1], size=num_days, p=[0.95, 0.05])  # 5% failure rate

# Create DataFrame
data = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperatures,
    "failure": failures
})

# Save to CSV
data.to_csv("temperature_data.csv", index=False)
print("Temperature data saved to 'temperature_data.csv'.")

