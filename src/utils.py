import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(trainer, save_path=None):

    epochs = np.arange(1, len(trainer.MSE_train) + 1)

    plt.style.use("seaborn-darkgrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- MSE ----
    axes[0].plot(epochs, trainer.MSE_train, label="Train", linewidth=2)
    axes[0].plot(epochs, trainer.MSE_val, label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Mean Squared Error")
    axes[0].legend()

    # ---- PCC ----
    axes[1].plot(epochs, trainer.PCC_train, label="Train", linewidth=2)
    axes[1].plot(epochs, trainer.PCC_val, label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PCC")
    axes[1].set_title("Pearson Correlation Coefficient")
    axes[1].legend()

    fig.suptitle("Training Curves", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
