import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "readings": [45, 50, 55, 60, 58, 57, 56, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38],
    "timestamp": "2024-12-17T12:00:00Z",
}
response = requests.post(url, json=payload)
print(response.json())

