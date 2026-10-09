import os

import torch
from torch.utils.data import DataLoader

from aria.data.image_class_mapping import PreprocessedDatasetWithPaths
from aria.models.spatial_AE import SpatialAutoencoder
from aria.utils.vae_plot import (
    collect_latent_coordinates,
    plot_tsne_with_dbscan,
)


def load_model(model_path, latent_channels, variational, device) -> SpatialAutoencoder:
    """Load the pre-trained spatial autoencoder model."""

    autoencoder = SpatialAutoencoder(latent_channels=latent_channels, variational=variational)
    state_dict = torch.load(model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)
    autoencoder.eval()

    return autoencoder


def prepare_data_loader(preprocessed_file: str, batch_size: int) -> DataLoader:
    """Load the preprocessed dataset and create a DataLoader."""

    dataset = PreprocessedDatasetWithPaths(preprocessed_file)
    processed_data = [(img.to(torch.float32), label, path) for img, label, path in dataset]
    return DataLoader(processed_data, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    # point this at the run directory written by aria/scripts/pre_processed_test.py
    run_dir = "/Users/jprice/Documents/aria/Autoencoder_work/flat_k_IQ_20261009_141629"
    model_path = os.path.join(run_dir, "spatial_vae_4_512.pth")
    preprocessed_file = "/Users/jprice/Documents/aria/preprocessed_image_tensors/IQtester_512.pt"
    latent_channels = 4
    variational = True
    batch_size = 8
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    autoencoder = load_model(model_path, latent_channels, variational, device)
    data_loader = prepare_data_loader(preprocessed_file, batch_size)

    # Collect latent coords and save
    input_csv_path = os.path.join(run_dir, "latent_coordinates_spatial_512.csv")
    collect_latent_coordinates(
        autoencoder,
        data_loader,
        device=device,
        num_batches=200,
        output_csv=input_csv_path,
        pool_size=8,
    )

    # eps was tuned for the old 16-d latent; expect to retune it for the new embedding
    plot_tsne_with_dbscan(
        input_csv_path,
        os.path.join(run_dir, "tsne_dbscan.csv"),
        num_dimensions=2,
        perplexity=20,
        n_iter=1000,
        eps=3.7,
        min_samples=10,
        save_csv=True,
    )
