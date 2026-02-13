import torch
from tqdm import tqdm


def train(model, data_loader, epochs, device):
    """Train the autoencoder model with a more granular progress bar.

    Args:
        model: The autoencoder model.
        data_loader: DataLoader for the training data.
        epochs: Number of epochs to train for.
        device: Device to train on (CPU or GPU).

    Returns:
        Trained model.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()  # Mean Squared Error for reconstruction

    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"Epoch [{epoch + 1}/{epochs}]")

        # Create progress bar for this epoch
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}", leave=True) as pbar:
            for batch_idx, batch in enumerate(data_loader):
                # Move data to device
                patches = batch.view(-1, 1, 512, 512).to(device)

                # Forward pass
                optimizer.zero_grad()
                reconstructed_patches = model(patches)[0]

                # Loss and backward pass
                loss = criterion(reconstructed_patches, patches)
                loss.backward()
                optimizer.step()

                # Update epoch loss
                epoch_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"batch_loss": loss.item(), "epoch_loss": epoch_loss})
                pbar.update(1)

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}")

    return model
