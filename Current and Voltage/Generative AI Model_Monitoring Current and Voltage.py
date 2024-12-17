import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Custom Dataset for current and voltage
class CurrentVoltageDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# Define the Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

# Loss function for VAE (Reconstruction + KL Divergence)
def vae_loss(reconstructed, original, mu, log_var):
    recon_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence

# Generate synthetic current and voltage data
def generate_synthetic_data(model, latent_dim, num_samples=1000):
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        generated_data = model.decoder(z).numpy()
    return generated_data

# Load and preprocess the data
def preprocess_data():
    # Example: Simulated current (A) and voltage (V) data
    time = np.linspace(0, 10, 1000)
    current = 5 + 0.5 * np.sin(2 * np.pi * time) + np.random.normal(0, 0.1, len(time))
    voltage = 220 + 10 * np.sin(2 * np.pi * time / 2) + np.random.normal(0, 2, len(time))
    
    # Combine current and voltage into one dataset
    data = np.column_stack((current, voltage))
    return data

# Train the VAE
def train_vae(data, input_dim, latent_dim, epochs=50, batch_size=64, learning_rate=0.001):
    dataset = CurrentVoltageDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed, mu, log_var = model(batch)
            loss = vae_loss(reconstructed, batch, mu, log_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    
    return model

# Main script
if __name__ == "__main__":
    # Preprocess data
    data = preprocess_data()

    # Hyperparameters
    input_dim = data.shape[1]  # 2 (current, voltage)
    latent_dim = 2  # Compress into 2 latent variables
    epochs = 50
    batch_size = 64
    learning_rate = 0.001

    # Train VAE
    print("Training Variational Autoencoder...")
    model = train_vae(data, input_dim, latent_dim, epochs, batch_size, learning_rate)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_data = generate_synthetic_data(model, latent_dim, num_samples=500)
    
    # Plot original and synthetic data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data[:, 0], label="Current (Original)", alpha=0.7)
    plt.plot(data[:, 1], label="Voltage (Original)", alpha=0.7)
    plt.legend()
    plt.title("Original Data")
    
    plt.subplot(1, 2, 2)
    plt.plot(synthetic_data[:, 0], label="Current (Synthetic)", alpha=0.7)
    plt.plot(synthetic_data[:, 1], label="Voltage (Synthetic)", alpha=0.7)
    plt.legend()
    plt.title("Synthetic Data")
    plt.tight_layout()
    plt.show()

