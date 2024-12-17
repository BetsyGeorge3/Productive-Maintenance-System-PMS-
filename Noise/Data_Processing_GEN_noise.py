import pyaudio
import numpy as np
import librosa

# Parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1             # Mono sound
RATE = 16000             # Sample rate (16 kHz)
CHUNK = 1024             # Number of frames per buffer
DEVICE_INDEX = 0         # Device index (select your microphone)

# Initialize PyAudio for microphone
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)

# Function to extract MFCC features from audio data
def extract_mfcc(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Function to get real-time audio data
def get_audio_data():
    audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    return audio_data / np.max(np.abs(audio_data))  # Normalize the audio data

# Collect data and preprocess it in real-time
def monitor_audio():
    while True:
        audio_data = get_audio_data()
        mfcc_features = extract_mfcc(audio_data)
        mfcc_features = np.reshape(mfcc_features, (1, 13))  # Adjust shape based on model input
        reconstructed_data = vae_model.predict(mfcc_features)
        
        # Calculate reconstruction error for anomaly detection
        error = np.mean(np.square(mfcc_features - reconstructed_data))
        
        if error > THRESHOLD:  # Set a threshold for anomaly detection
            print("Anomaly Detected: Abnormal Noise!")
        else:
            print("Normal Noise")

# Start monitoring
monitor_audio()

# Load your dataset (features extracted from audio)
X_train = np.load('normal_noise_features.npy')  # This should be MFCC features extracted from your audio data

# Train the model
vae_model.fit(X_train, X_train, epochs=20, batch_size=64)

# Save the model for deployment
vae_model.save('noise_detection_vae_model.h5')

