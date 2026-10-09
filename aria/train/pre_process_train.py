import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm

from aria.train.losses import reconstruction_loss, ssim


def load_preprocessed_data(file_path):
    preprocessed_data = torch.load(file_path)
    if isinstance(preprocessed_data, list) and isinstance(preprocessed_data[0], tuple):
        print("Preprocessed dataset loaded successfully.")
    else:
        raise TypeError("Preprocessed dataset format is incorrect. Expected a list of (tensor, label) tuples.")
    return preprocessed_data


def create_data_loader(preprocessed_data, batch_size=32, val_fraction=0.0, seed=0):
    """Convert all image tensors to float32 and build DataLoaders.

    Returns a single train loader if ``val_fraction`` is 0, otherwise ``(train_loader, val_loader)``.
    """
    processed_data = [(img.to(torch.float32), label, path) for img, label, path in preprocessed_data]
    if val_fraction <= 0:
        return DataLoader(processed_data, batch_size=batch_size, shuffle=True)

    n_val = max(1, int(len(processed_data) * val_fraction))
    train_set, val_set = random_split(
        processed_data, [len(processed_data) - n_val, n_val], generator=torch.Generator().manual_seed(seed)
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
    )


def _images(batch):
    if len(batch) not in (2, 3):
        raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")
    return batch[0]


@torch.no_grad()
def evaluate(autoencoder, data_loader, device):
    autoencoder.eval()
    total_loss, total_ssim = 0.0, 0.0
    for batch in data_loader:
        x = _images(batch).to(device)
        x_hat = autoencoder(x)
        total_loss += reconstruction_loss(x_hat, x).item()
        total_ssim += ssim(x_hat, x).item()
    autoencoder.train()
    return total_loss / len(data_loader), total_ssim / len(data_loader)


@torch.no_grad()
def save_reconstructions(autoencoder, x, path):
    """Saves a grid with originals on the top row and reconstructions below."""
    autoencoder.eval()
    x_hat = autoencoder(x)
    autoencoder.train()
    save_image(torch.cat([x, x_hat]).cpu(), path, nrow=x.shape[0])


def train(
    autoencoder,
    data_loader,
    epochs=20,
    device="cpu",
    logger=None,
    lr=2e-4,
    kl_weight=1e-3,
    val_loader=None,
    sample_dir=None,
    sample_every=10,
    hflip=True,
):
    autoencoder.to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    autoencoder.train()

    sample_batch = None
    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
        sample_batch = _images(next(iter(val_loader or data_loader)))[:4].to(device)

    for epoch in range(epochs):
        total_loss = 0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            x = _images(batch).to(device)
            if hflip:
                # left/right captures are mirror images of each other, so flipping is a free 2x on data
                flip = torch.rand(x.shape[0], device=device) < 0.5
                x = torch.where(flip[:, None, None, None], x.flip(-1), x)

            optimizer.zero_grad()
            x_hat = autoencoder(x)
            loss = reconstruction_loss(x_hat, x) + kl_weight * getattr(autoencoder.encoder, "kl", 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        scheduler.step()

        epoch_loss = total_loss / len(data_loader)
        if logger:
            logger.log_epoch_loss(epoch_loss)

        msg = f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}"
        if val_loader is not None:
            val_loss, val_ssim = evaluate(autoencoder, val_loader, device)
            msg += f", Val loss: {val_loss:.4f}, Val SSIM: {val_ssim:.4f}"
        print(msg)

        if sample_batch is not None and ((epoch + 1) % sample_every == 0 or epoch + 1 == epochs):
            save_reconstructions(autoencoder, sample_batch, os.path.join(sample_dir, f"epoch_{epoch+1:03d}.png"))

    # Save the metrics after training
    if logger:
        logger.save()
        logger.plot_loss_vs_epochs()

    return autoencoder
