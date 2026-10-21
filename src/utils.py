"""Shared helpers: reproducibility, checkpoints, plotting."""
import os
import random

import matplotlib
matplotlib.use("Agg")  # save plots to file without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------- Reproducibility ----------
def set_seed(seed: int) -> None:
    """Fix every random source so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own fixed seed (for random augmentations)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """Seeded generator so DataLoader shuffling order is repeatable."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------- Checkpoints ----------
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_acc, cfg) -> None:
    """Save everything needed to reload the model or continue training."""
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "config": cfg.to_dict(),
    }, path)


def load_checkpoint(path, model, device):
    """Load model weights from a checkpoint. Returns the full checkpoint dict."""
    # weights_only=False is safe here: we only load files we created ourselves
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# ---------- Plotting ----------
def plot_curves(history: list, out_path: str, title: str) -> None:
    """Plot train vs validation loss and accuracy per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] * 100 for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] * 100 for h in history], label="validation")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
