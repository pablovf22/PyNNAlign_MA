import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(trainer, save_path=None):

    epochs = np.arange(1, len(trainer.MSE_train) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

    # ---- MSE ----
    ax = axes[0]
    ax.plot(epochs, trainer.MSE_train, label="Train", linewidth=2)
    ax.plot(epochs, trainer.MSE_val, label="Validation", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)

    # ---- PCC ----
    ax = axes[1]
    ax.plot(epochs, trainer.PCC_train, label="Train", linewidth=2)
    ax.plot(epochs, trainer.PCC_val, label="Validation", linewidth=2)
    ax.set_title("PCC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PCC")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)

    fig.suptitle("Training Curves", y=1.02, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)