import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def preprocess_16bit_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> torch.Tensor | None:
    """Reads a 16-bit grayscale image, normalizes it while preserving dynamic range, and resizes it to match the model input size.

    :param image_path: Path to the image file.
    :param target_size: Desired size of the output image.
    :return: Processed image as a float32 torch.Tensor or None if the image could not be loaded.
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

    # percentile returns float64 scalars, which would otherwise promote the whole image to float64
    return torch.from_numpy(image.astype(np.float32)).unsqueeze(0)


def find_images(image_dir: str, patterns: list[str] | None = None) -> list[str]:
    """Recursively finds image files in ``image_dir`` whose filename contains any of ``patterns``.

    Matching is a plain substring check, e.g. ``"black_5.0000s"``.
    """
    matches = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            if patterns and not any(p in file for p in patterns):
                continue
            matches.append(os.path.join(root, file))
    return sorted(matches)


def images_to_tensor_file(
    image_dir: str, output_file: str, patterns: list[str] | None = None, target_size: tuple[int, int] = (512, 512)
) -> int:
    """Preprocesses matching images and saves them as a list of (tensor, label, path) tuples.

    :return: Number of images saved.
    """
    image_paths = find_images(image_dir, patterns)
    if not image_paths:
        raise FileNotFoundError(f"No images in {image_dir} with filenames containing {patterns}")

    preprocessed_data = []
    for img_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        img = preprocess_16bit_image(img_path, target_size=target_size)
        if img is None:
            continue
        label = 0
        preprocessed_data.append((img, label, img_path))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    torch.save(preprocessed_data, output_file)
    print(f"Saved {len(preprocessed_data)} of {len(image_paths)} images to {output_file}")
    return len(preprocessed_data)


if __name__ == "__main__":
    input_dir = "/Users/jprice/Documents/Pupil_Ghost/DOE_captures"
    output_dir = "/Users/jprice/Documents/aria/preprocessed_image_tensors"
    suffix = "short"  # keep files whose name contains this
    image_size = 512

    output_file = os.path.join(output_dir, f"{suffix}_{image_size}.pt")
    images_to_tensor_file(input_dir, output_file, patterns=[suffix], target_size=(image_size, image_size))
