import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# image_class_mapping.py

import os
import random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from aria.data.image_class_mapping import CustomImageDataset, PreprocessedDatasetWithPaths


def preprocess_16bit_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> torch.Tensor | None:
    """Reads a 16-bit grayscale image, normalizes it while preserving dynamic range, and resizes it to match the model input size.

    :param image_path: Path to the image file.
    :param target_size: Desired size of the output image.
    :return: Processed image as a torch.Tensor or None if the image could not be loaded.
    """

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Warning: Could not load image {image_path}. Skipping.")
        return None

    # Normalize the image and stretch contrast
    if image.dtype == np.uint16:
        image = image.astype(np.float32)
        image /= 65535.0
        min_val, max_val = np.percentile(image, (0, 100))
        image = np.clip((image - min_val) / (max_val - min_val + 1e-6), 0, 1)
    else:
        image = image.astype(np.float32) / 255.0

    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Convert to tensor bit
    return torch.from_numpy(image).unsqueeze(0)


if __name__:
    image_dir = r'/Users/jprice/Documents/aria/preprocessed_images/IQ TESTER'
    output_file = r"/Users/jprice/Documents/aria/preprocessed_image_tensors/IQtester_512.pt"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset = CustomImageDataset(root_dir=image_dir, transform=None)
    preprocessed_data = []

    print("Preprocessing images with augmentations...")

    for i in tqdm(range(len(dataset)), desc="Processing Images", unit="image"):
        img_path = dataset.image_paths[i]
        img = preprocess_16bit_image(img_path, target_size=(512, 512))

        if img is None:
            continue

        label = 0
        preprocessed_data.append((img, label, img_path))

    # Save to .pt file
    torch.save(preprocessed_data, output_file)
    print(f"Preprocessed dataset with metadata saved to {output_file}")
