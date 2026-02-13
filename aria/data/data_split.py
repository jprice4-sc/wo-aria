import os
import random
import shutil

from sklearn.model_selection import train_test_split


def split_data(src_dir, output_dir, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    """Splits the data into train, validation, and test sets.
    - src_dir: Source directory containing the dataset.
    - output_dir: Output directory to save the split dataset.
    - train_size: Proportion of data to use for training.
    - val_size: Proportion of data to use for validation.
    - test_size: Proportion of data to use for testing.
    - seed: Random seed for reproducibility.
    """
    assert train_size + val_size + test_size == 1, "train_size + val_size + test_size must equal 1"

    random.seed(seed)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for cls in classes:
        cls_dir = os.path.join(src_dir, cls)
        images = os.listdir(cls_dir)
        train_and_val_images, test_images = train_test_split(images, test_size=test_size, random_state=seed)
        train_images, val_images = train_test_split(
            train_and_val_images, test_size=val_size / (train_size + val_size), random_state=seed
        )

        for category, category_images in zip(
            ["train", "val", "test"], [train_images, val_images, test_images], strict=False
        ):
            category_dir = os.path.join(output_dir, category, cls)
            os.makedirs(category_dir, exist_ok=True)
            for img in category_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(category_dir, img)
                shutil.copyfile(src, dst)


if __name__ == "__main__":
    src_dir = "/Users/svc-wolab-01/Documents/ADA/curated_dataset"
    output_dir = "/Users/svc-wolab-01/Documents/ADA/split_dataset"
    split_data(src_dir, output_dir)
