# data_loader.py
import os
from pathlib import Path

from google.cloud import storage
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def create_dataloaders(train_dir: str, test_dir: str, batch_size: int, transform=None) -> tuple:
    """Function to loaf data for model training."""
    if transform is None:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
