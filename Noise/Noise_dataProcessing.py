import pyaudio
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model

# Parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1             # Mono sound
RATE = 16000             # Sample rate (16 kHz)
CHUNK = 1024             # Number of frames per buffer
DEVICE_INDEX = 0         # Device index (select your microphone)
THRESHOLD = 0.5          # Anomaly detection threshold (for demo purposes)

# Initialize PyAudio for microphone
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)

# Load your trained model (assuming you have a pre-trained anomaly detection model)
model = load_model('path_to_your_trained_model.h5')  # Replace with your model path

# Function to extract features from the audio (MFCC)
def extract_features(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Function to monitor real-time audio
def monitor_audio():
    print("Monitoring audio...")

    while True:
        # Read audio data from microphone
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        # Preprocess and extract features (MFCC)
        audio_data_normalized = audio_data / np.max(np.abs(audio_data))  # Normalize the audio data
        features = extract_features(audio_data_normalized)
        
        # Reshape features for model input
        features_reshaped = np.reshape(features, (1, 13))  # Adjust shape according to your model

        # Predict whether it's normal or abnormal
        prediction = model.predict(features_reshaped)

        # Check if the prediction indicates abnormal behavior (using a threshold)
        if prediction >= THRESHOLD:
            print("ALERT: Abnormal noise detected!")
        else:
            print("Noise is normal")

# Start monitoring the microphone
monitor_audio()

