import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from aria.data.image_class_mapping import CustomImageDataset, PreprocessedDatasetWithPaths
from aria.models.auto_encoder import Autoencoder
from aria.models.conv_AE import ConvAutoencoder
from aria.utils.vae_plot import (
    collect_latent_coordinates,
    plot_reconstructed,
    plot_tsne_with_dbscan,
)


def load_model(model_path, latent_dims, device) -> Autoencoder:
    """Load the pre-trained autoencoder model."""

    autoencoder = ConvAutoencoder(latent_dims)
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
    model_path = r"D:\ADA\Proto_A_data\Autoencoder_work\flat_k_IQ_20250417_103206\autoencoder_16_512.pth"
    preprocessed_file = r"D:\ADA\Proto_A_data\Autoencoder_work\preprocessed image tensors\preprocessed_IQtester_512.pt"
    latent_dims = 16
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = load_model(model_path, latent_dims, device)
    data_loader = prepare_data_loader(preprocessed_file, batch_size)

    # Plot reconstructed images
    # plot_reconstructed(autoencoder, latent_dims, device=device, image_size=224)

    # Collect latent coords and save
    collect_latent_coordinates(
        autoencoder,
        data_loader,
        device=device,
        num_batches=200,
        output_csv=r"D:\ADA\Proto_A_data\Autoencoder_work\flat_k_AA_20250417_094507\latent_coordinates_conAE_512.csv",
    )

    input_csv_path = r"D:\ADA\Proto_A_data\Autoencoder_work\flat_k_AA_20250417_094507\latent_coordinates_conAE_512.csv"
    output_csv_path = r"D:\ADA\Proto_A_data\Autoencoder_work\flat_k_AA_20250417_094507\tsne_dbscan.csv"

    plot_tsne_with_dbscan(
        input_csv_path,
        output_csv_path,
        num_dimensions=2,
        perplexity=20,
        n_iter=1000,
        eps=3.7,
        min_samples=10,
        save_csv=True,
    )
