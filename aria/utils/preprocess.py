import os

import numpy as np
from PIL import Image


def process_original_image(image_path, target_size=(224, 224)):
    """Converts original 16-bit image to grayscale, resizes it, normalizes pixel values, and applies contrast stretching."""
    image = Image.open(image_path).convert("I")
    image = image.resize(target_size, Image.LANCZOS)
    image_np = np.array(image, dtype=np.float32)
    image_np /= 65535.0

    min_val, max_val = np.percentile(image_np, (0, 100))
    image_np = np.clip((image_np - min_val) / (max_val - min_val + 1e-6), 0, 1)

    return image_np


def process_images_in_folder(root_folder, output_folder, target_size=(512, 683)):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for subdir, _, files in os.walk(root_folder):
        # Skip processing if 'final_test' is not in the subdir path
        if "final_test" not in subdir:
            continue

        for file in files:
            if "green_black_5.00000s" in file.lower() and file.lower().endswith((".png", "jpg", "jpeg", "bmp", "tiff")):
                file_path = os.path.join(subdir, file)

                # Determine the serial number directory by going one level up from 'final_test'
                serial_number_dir = os.path.dirname(subdir)
                serial_number = os.path.basename(serial_number_dir)

                # Prepare the output file path without the 'final_test' subdirectory
                output_subdir = os.path.join(output_folder, serial_number)
                os.makedirs(output_subdir, exist_ok=True)
                output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + ".png")

                # Skip if the output file already exists
                if os.path.exists(output_file_path):
                    print(f"Skipping {file_path} as the output file already exists.")
                    continue

                try:
                    print(f"Processing {file_path}")
                    processed_image = process_original_image(file_path, target_size)

                    # Convert back to an image and save
                    processed_image_uint16 = (processed_image * 65535).astype(np.uint16)
                    processed_img_pil = Image.fromarray(processed_image_uint16)

                    # Save the image as PNG
                    processed_img_pil.save(output_file_path, "PNG")
                except Exception as e:
                    print(f"Skipping {file_path} due to an error: {e}")


# Example usage
root_folder = r"D:\ADA\Test_flow_cleanup\IQ TESTER"
output_folder = r"D:\ADA\Test_flow_cleanup\IQ_test_preprocessed"
process_images_in_folder(root_folder, output_folder)
