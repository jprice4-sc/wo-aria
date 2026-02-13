# vae_training.py
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(autoencoder, data_loader, epochs=20, device="cpu"):
    autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters())
    autoencoder.train()

    for epoch in range(epochs):
        total_loss = 0
        # Use tqdm to wrap the data_loader and display a progress bar
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x, _ in progress_bar:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = autoencoder(x)
            loss = F.mse_loss(x_hat, x) + getattr(autoencoder.encoder, "kl", 0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}")

    return autoencoder
