import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

from aria.models.auto_encoder import Autoencoder
from aria.models.conv_AE import ConvAutoencoder

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


def process_original_image(image_path, target_size=(224, 224)):
    """Converts original 16-bit image to grayscale, resizes it to match the autoencoder output,
    normalizes pixel values properly, and applies contrast stretching.
    """
    image = Image.open(image_path).convert("I")  # Preserve 16-bit depth
    image = image.resize(target_size, Image.LANCZOS)  # Resize to match reconstruction

    # normalisation
    image_np = np.array(image).astype(np.float32)
    image_np = image_np / 65535.0
    min_val, max_val = np.percentile(image_np, (2, 98))
    image_np = np.clip((image_np - min_val) / (max_val - min_val + 1e-6), 0, 1)

    return image_np


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
