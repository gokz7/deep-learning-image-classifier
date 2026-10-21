"""Evaluate a trained model on the TEST set (the only place the test set is used).
Usage: python -m src.evaluate --model baseline
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.config import Config, CLASSES
from src.data import get_dataloaders, denormalize
from src.models import get_model, count_params
from src.utils import set_seed, load_checkpoint


@torch.no_grad()
def predict(model, loader, device):
    """Return class probabilities, true labels and images for a whole dataset."""
    model.eval()
    all_probs, all_labels, all_images = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
        all_labels.append(labels)
        all_images.append(images)
    return torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_images)


def save_confusion_matrix(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_misclassified(images, labels, preds, confs, out_path, title, n=16):
    """Show the n most confident wrong predictions."""
    wrong = (preds != labels).nonzero(as_tuple=True)[0]
    wrong = wrong[confs[wrong].argsort(descending=True)][:n]

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for ax, idx in zip(axes.flat, wrong.tolist()):
        ax.imshow(denormalize(images[idx]).permute(1, 2, 0))
        ax.set_title(f"true: {CLASSES[labels[idx]]}\npred: {CLASSES[preds[idx]]} ({confs[idx]*100:.0f}%)",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = cfg.device
    out_dir = os.path.join(cfg.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    # Load best checkpoint (chosen by validation accuracy during training)
    model = get_model(args.model, cfg).to(device)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"best_{args.model}.pt")
    ckpt = load_checkpoint(ckpt_path, model, device)
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val acc {ckpt['best_val_acc']*100:.2f}%)")

    # Test set: used here only
    _, _, test_loader = get_dataloaders(cfg)
    probs, labels, images = predict(model, test_loader, device)
    confs, preds = probs.max(dim=1)

    # Overall and per-class accuracy
    test_acc = (preds == labels).float().mean().item()
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(len(CLASSES))))
    per_class = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)}

    # Top 5 confused pairs (true -> predicted)
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    top_pairs = []
    for flat_idx in off_diag.ravel().argsort()[::-1][:5]:
        t, p = divmod(int(flat_idx), len(CLASSES))
        top_pairs.append({"true": CLASSES[t], "pred": CLASSES[p], "count": int(off_diag[t, p])})

    # Print results
    print(f"\nTEST ACCURACY: {test_acc*100:.2f}%  ({int((preds == labels).sum())}/{len(labels)} correct)")
    print("\nPer-class accuracy:")
    for c, a in sorted(per_class.items(), key=lambda x: x[1]):
        print(f"  {c:12s} {a*100:6.2f}%")
    print("\nMost common mistakes (true -> predicted):")
    for p in top_pairs:
        print(f"  {p['true']:10s} -> {p['pred']:10s} {p['count']} images")

    # Save plots
    save_confusion_matrix(cm, os.path.join(out_dir, "confusion_matrix.png"),
                          f"{args.model} — confusion matrix (test acc {test_acc*100:.2f}%)")
    save_misclassified(images, labels, preds, confs, os.path.join(out_dir, "misclassified.png"),
                       f"{args.model} — most confident mistakes on the test set")

    # Save metrics (merge with training summary if it exists)
    metrics = {}
    summary_path = os.path.join(out_dir, "train_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            metrics.update(json.load(f))
    metrics.update({
        "model": args.model,
        "trainable_params": count_params(model),
        "checkpoint_epoch": ckpt["epoch"],
        "best_val_acc": ckpt["best_val_acc"],
        "test_acc": test_acc,
        "per_class_test_acc": per_class,
        "top_confusions": top_pairs,
    })
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {out_dir}/metrics.json, confusion_matrix.png, misclassified.png")


if __name__ == "__main__":
    main()
