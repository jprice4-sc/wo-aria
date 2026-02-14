# image_class_mapping.py

import os
import random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# Ensure loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if image.mode != "L":
            image = image.convert("L")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Assuming you don't have labels

class PreprocessedDatasetWithPaths(torch.utils.data.Dataset):
    def __init__(self, preprocessed_file):
        self.data = torch.load(preprocessed_file)
        self.image_paths = [item[2] for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the tuple, expected to contain (image tensor, label, image path)
        return self.data[idx]
