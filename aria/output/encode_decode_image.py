import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from aria.models.auto_encoder import Autoencoder
from aria.utils.preprocess import process_original_image

img_size = 224


def load_model(model_path, latent_dims, device):
    """Loads the trained autoencoder model from the specified path."""
    model = torch.load(model_path, map_location=device)

    autoencoder = Autoencoder(latent_dims, image_size=img_size).to(device)
    autoencoder.load_state_dict(model)

    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def prepare_image(image_path, device):
    """Prepares the image for inference by resizing and normalizing it."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Read image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = Image.fromarray(image).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    return image


def visualize_reconstruction(original_corrected, reconstructed, title="Reconstruction"):
    """Displays:
    - The original image with optimized 16-bit contrast correction.
    - The reconstructed image directly as-is.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # plots
    axes[0].imshow(original_corrected, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original (16-bit Corrected)")
    axes[0].axis("off")

    axes[1].imshow(reconstructed, cmap="gray", vmin=0.7, vmax=1.1)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_path = r"D:\ADA\AE's\saved_models\autoencoder_16_224.pth"
    image_path = r"D:\ADA\AE's\aa_projector_data\flat_k\left_flat_k\TF20AS1_Left_flat_k.png"
    latent_dims = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    autoencoder = load_model(model_path, latent_dims, device)
    image = prepare_image(image_path, device)

    # Get the reconstruction
    with torch.no_grad():
        reconstructed = autoencoder(image)
        reconstructed = reconstructed.squeeze().cpu().numpy()

    original_corrected = process_original_image(image_path, target_size=(img_size, img_size))

    # Visualize
    visualize_reconstruction(original_corrected, reconstructed)
