import json
import os

import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, log_dir="logs", log_file="training_metrics.json"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.metrics = {
            "epoch_losses": [],
        }
        self.log_dir = log_dir

    def log_epoch_loss(self, epoch_loss):
        self.metrics["epoch_losses"].append(epoch_loss)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def plot_loss_vs_epochs(self):
        epochs = range(1, len(self.metrics["epoch_losses"]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics["epoch_losses"], marker="o", linestyle="-")
        plt.title("Training Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        # Save the plot in the same directory as the log file
        plot_path = os.path.join(self.log_dir, "loss_vs_epochs.png")
        plt.savefig(plot_path)
        plt.close()
