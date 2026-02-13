import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aria.models.auto_encoder import Autoencoder


def load_preprocessed_data(file_path):
    preprocessed_data = torch.load(file_path)
    if isinstance(preprocessed_data, list) and isinstance(preprocessed_data[0], tuple):
        print("Preprocessed dataset loaded successfully.")
    else:
        raise TypeError("Preprocessed dataset format is incorrect. Expected a list of (tensor, label) tuples.")
    return preprocessed_data


def create_data_loader(preprocessed_data, batch_size=32):
    """Convert all image tensors to float32 before creating DataLoader."""

    processed_data = [(img.to(torch.float32), label, path) for img, label, path in preprocessed_data]
    return DataLoader(processed_data, batch_size=batch_size, shuffle=True)


def train(autoencoder, data_loader, epochs=20, device="cpu", logger=None, lr=1e-3):
    autoencoder.to(device)
    optimizer = torch.optim.Adagrad(autoencoder.parameters(), lr=lr)
    autoencoder.train()

    for epoch in range(epochs):
        total_loss = 0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            if len(batch) == 2:
                x, _ = batch
            elif len(batch) == 3:
                x, _, _ = batch
            else:
                raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")

            x = x.to(device)
            optimizer.zero_grad
            x_hat = autoencoder(x)
            loss = torch.nn.functional.mse_loss(x_hat, x) + getattr(
                autoencoder.encoder, "kl", 0
            )  # Add KL divergence if present
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Log the epoch loss
        if logger:
            logger.log_epoch_loss(total_loss / len(data_loader))

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}")

    # Save the metrics after training
    if logger:
        logger.save()
        logger.plot_loss_vs_epochs()

    return autoencoder
