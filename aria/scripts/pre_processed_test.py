import os
import time
from datetime import datetime

import torch

from aria.models.spatial_AE import SpatialAutoencoder
from aria.train.pre_process_train import create_data_loader, load_preprocessed_data, train
from aria.utils.metric_logger import MetricLogger


def main():
    start = time.time()
    preprocessed_file = "/Users/jprice/Documents/aria/preprocessed_image_tensors/short_512.pt"
    preprocessed_data = load_preprocessed_data(preprocessed_file)

    latent_channels = 4  # latent is latent_channels x 64 x 64 for 512 x 512 inputs
    variational = True  # needed if you want to sample new images; set False for pure reconstruction
    kl_weight = 1e-3
    batch_size = 8
    image_size = 512
    epochs = 25
    lr = 2e-4
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    autoencoder = SpatialAutoencoder(latent_channels=latent_channels, variational=variational)
    train_loader, val_loader = create_data_loader(preprocessed_data, batch_size=batch_size, val_fraction=0.1)

    # timestamp and dir stuff
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/Users/jprice/Documents/aria/Autoencoder_work/flat_k_IQ_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # metric logger
    logger = MetricLogger(log_dir=save_dir)

    trained_autoencoder = train(
        autoencoder,
        train_loader,
        epochs=epochs,
        device=device,
        logger=logger,
        lr=lr,
        kl_weight=kl_weight,
        val_loader=val_loader,
        sample_dir=os.path.join(save_dir, "samples"),
    )

    model_save_path = os.path.join(
        save_dir, f"spatial_{'vae' if variational else 'ae'}_{latent_channels}_{image_size}.pth"
    )
    torch.save(trained_autoencoder.state_dict(), model_save_path)

    print(f"Model and metrics saved successfully in {save_dir}.")
    print(f"Time taken: {time.time() - start} seconds.")


if __name__ == "__main__":
    main()
