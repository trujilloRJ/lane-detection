import numpy as np
import matplotlib.pyplot as plt
from common import extract_log_data

if __name__=="__main__":
    checkpoint_folder = "checkpoints"
    exp_name = "sUNetWide_v8_Srop_adam"
    log_file_path = f"{checkpoint_folder}/{exp_name}.log"
    epochs, train_losses, validation_losses = extract_log_data(log_file_path)

    min_val_loss = np.min(validation_losses)
    best_epoch = epochs[np.argmin(validation_losses)]

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, validation_losses, label="validation", zorder=0)
    ax.scatter(best_epoch, min_val_loss, color='g', s=20, label=f"epoch {best_epoch} -> {min_val_loss:.3f}", zorder=1)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{checkpoint_folder}/{exp_name}.png")