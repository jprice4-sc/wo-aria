import matplotlib.pyplot as plt
import torch

from aria.models.spatial_AE import SpatialAutoencoder
from aria.train.losses import ssim
from aria.utils.image2tensor import preprocess_16bit_image

img_size = 512


def load_model(model_path, latent_channels, variational, device):
    """Loads the trained spatial autoencoder model from the specified path."""
    autoencoder = SpatialAutoencoder(latent_channels=latent_channels, variational=variational)
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def prepare_image(image_path, device):
    """Prepares the image for inference using the same preprocessing as training."""
    image = preprocess_16bit_image(image_path, target_size=(img_size, img_size))
    if image is None:
        raise ValueError(f"Could not load image {image_path}")
    return image.unsqueeze(0).to(device=device, dtype=torch.float32)


def visualize_reconstruction(original, reconstructed, title="Reconstruction"):
    """Displays the original, the reconstruction and their absolute difference on the same intensity scale."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")

    axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(title)

    # difference is small, so let it autoscale to make residual structure visible
    diff = axes[2].imshow(abs(original - reconstructed), cmap="magma")
    axes[2].set_title("|Original - Reconstruction|")
    fig.colorbar(diff, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # point this at the .pth saved by aria/scripts/pre_processed_test.py
    model_path = "/Users/jprice/Documents/aria/Autoencoder_work/flat_k_IQ_20261009_141629/spatial_vae_4_512.pth"
    image_path = "/Users/jprice/Documents/ADA_beta/DVT/DOE9/black_IQ_TESTER/000004943_C26010292A7_left_final_test_green_black_0.99911s.png"
    latent_channels = 4
    variational = True
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    autoencoder = load_model(model_path, latent_channels, variational, device)
    image = prepare_image(image_path, device)

    with torch.no_grad():
        reconstructed = autoencoder(image)
        print(f"SSIM: {ssim(reconstructed, image).item():.4f}")

    visualize_reconstruction(image.squeeze().cpu().numpy(), reconstructed.squeeze().cpu().numpy())
