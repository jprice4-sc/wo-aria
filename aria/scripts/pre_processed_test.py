import os
import time
from datetime import datetime

import torch

from ada.models.auto_encoder import Autoencoder
from ada.models.conv_AE import ConvAutoencoder
from ada.train.pre_process_train import create_data_loader, load_preprocessed_data, train
from ada.utils.metric_logger import MetricLogger


def main():
    start = time.time()
    preprocessed_file = (
        r"D:\ADA\Proto_A_data\Autoencoder_work\preprocessed image tensors\preprocessed_IQtester_fill_512.pt"
    )
    preprocessed_data = load_preprocessed_data(preprocessed_file)

    latent_dims = 16
    batch_size = 8
    image_size = 512
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    autoencoder = ConvAutoencoder(latent_dims)
    data_loader = create_data_loader(preprocessed_data, batch_size=batch_size)

    # timestamp and dir stuff
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"D:\\ADA\\Proto_A_data\\Autoencoder_work\\flat_k_IQ_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # metric logger
    logger = MetricLogger(log_dir=save_dir)

    trained_autoencoder = train(autoencoder, data_loader, epochs=100, device=device, logger=logger, lr=lr)

    model_save_path = os.path.join(save_dir, f"autoencoder_{latent_dims}_{image_size}.pth")
    torch.save(trained_autoencoder.state_dict(), model_save_path)

    print(f"Model and metrics saved successfully in {save_dir}.")
    print(f"Time taken: {time.time() - start} seconds.")


if __name__ == "__main__":
    main()
