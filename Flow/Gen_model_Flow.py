import tensorflow as tf
from tensorflow.keras import layers

# Define the generator model
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='tanh'))  # Output is a single flow value (scaled [-1,1])
    return model

# Define the discriminator model
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output: probability that data is real (0 or 1)
    return model

# Define the GAN by combining the generator and discriminator
def build_gan(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

# GAN Training Loop
def train_gan(generator, discriminator, gan, flow_data, epochs=10000, batch_size=128, latent_dim=100):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Train the discriminator on real and fake data
        idx = np.random.randint(0, flow_data.shape[0], half_batch)
        real_flow_data = flow_data[idx]
        
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_flow_data = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_flow_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_flow_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator to fool the discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # Print the progress
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

# Compile the models
latent_dim = 100
discriminator = build_discriminator((1,))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator(latent_dim)
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train the GAN
train_gan(generator, discriminator, gan, flow_data_scaled, epochs=10000, batch_size=64, latent_dim=latent_dim)

# Generate synthetic flow data
def generate_synthetic_flow_data(generator, num_samples=1000, latent_dim=100):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_flow_data = generator.predict(noise)
    
    # Rescale the data back to the original flow rate range
    return generated_flow_data

# Generate synthetic data
synthetic_flow_data = generate_synthetic_flow_data(generator)

# Rescale the data back to the original range
scaler = MinMaxScaler(feature_range=(0, 100))  # Original range of flow data
synthetic_flow_data_rescaled = scaler.fit_transform(synthetic_flow_data)

# Visualize the generated flow data
plt.plot(synthetic_flow_data_rescaled)
plt.title("Generated Flow Data")
plt.xlabel("Time")
plt.ylabel("Flow Rate (L/min)")
plt.show()

