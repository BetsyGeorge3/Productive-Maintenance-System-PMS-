import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import numpy as np
import pyaudio

# VAE model definition
def build_vae(input_shape):
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    z_mean = layers.Dense(32)(x)
    z_log_var = layers.Dense(32)(x)
    
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(32,))([z_mean, z_log_var])
    
    # Decoder
    decoder_hid = layers.Dense(64, activation='relu')
    decoder_upp = layers.Dense(128, activation='relu')
    decoder_out = layers.Dense(input_shape[0], activation='sigmoid')(decoder_upp(decoder_hid(z)))
    
    # VAE Model
    vae = models.Model(inputs, decoder_out)
    
    # Loss function
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, decoder_out))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    
    return vae

# Input shape for MFCCs (13 MFCC features for simplicity)
input_shape = (13,)  # Adjust based on your feature size
vae_model = build_vae(input_shape)

vae_model.compile(optimizer='adam')
vae_model.summary()

