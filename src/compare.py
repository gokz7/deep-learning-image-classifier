"""Compare all trained models using their saved results (no retraining, no re-testing).
Usage: python -m src.compare
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config

MODELS = ["baseline", "residual", "deep_plain", "deep_residual"]


def load_results(model, out_root):
    """Load metrics.json and history.csv for one model. Returns (None, None) if missing."""
    folder = os.path.join(out_root, model)
    metrics_path = os.path.join(folder, "metrics.json")
    history_path = os.path.join(folder, "history.csv")
    if not (os.path.exists(metrics_path) and os.path.exists(history_path)):
        return None, None
    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics, pd.read_csv(history_path)


def main():
    cfg = Config()
    rows, histories = [], {}

    for name in MODELS:
        m, h = load_results(name, cfg.output_dir)
        if m is None:
            print(f"Skipping {name}: results not found (train + evaluate it first)")
            continue
        histories[name] = h

        best_row = h.loc[h["val_acc"].idxmax()]
        high_lr = h[h["lr"] >= 0.5 * cfg.lr]          # epochs where LR >= 50% of peak
        per_class = m["per_class_test_acc"]

        rows.append({
            "Model": name,
            "Params": f"{m['trainable_params']:,}",
            "Best epoch": int(best_row["epoch"]),
            "Best val acc": f"{m['best_val_acc']*100:.2f}%",
            "Test acc": f"{m['test_acc']*100:.2f}%",
            "Val→test": f"{(m['test_acc'] - m['best_val_acc'])*100:+.2f}",
            "Train–val gap": f"{(best_row['train_acc'] - best_row['val_acc'])*100:.2f}",
            "High-LR val acc": f"{high_lr['val_acc'].mean()*100:.1f}% ± {high_lr['val_acc'].std()*100:.1f}",
            "Cat acc": f"{per_class['cat']*100:.1f}%",
            "Dog acc": f"{per_class['dog']*100:.1f}%",
            "Sec/epoch": f"{h['epoch_time_sec'].mean():.1f}",
        })

    if not rows:
        print("No results found.")
        return

    # ---- Markdown table ----
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in headers) + " |")
    high_lr_epochs = histories[rows[0]["Model"]].query(f"lr >= {0.5 * cfg.lr}")["epoch"]
    notes = [
        "",
        f"- All models: same data split, seed ({cfg.seed}), optimizer (SGD + OneCycle, peak LR {cfg.lr}), "
        f"batch size {cfg.batch_size}, {cfg.epochs} epochs.",
        "- Best epoch chosen by validation accuracy. The test set was used once per model, after training.",
        f"- High-LR phase = epochs {high_lr_epochs.min()}–{high_lr_epochs.max()} (LR ≥ 50% of peak). "
        "Value shown as mean ± standard deviation of validation accuracy.",
        "- Train–val gap is measured at the best epoch (train accuracy includes augmentation and dropout).",
    ]
    table_md = "# Model Comparison\n\n" + "\n".join(lines + notes) + "\n"
    out_md = os.path.join(cfg.output_dir, "comparison.md")
    with open(out_md, "w") as f:
        f.write(table_md)

    # ---- Overlay plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in histories.items():
        ax1.plot(h["epoch"], h["val_acc"] * 100, label=name)
        ax2.plot(h["epoch"], h["train_loss"], label=name)
    ax1.set_title("Validation accuracy (%)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylim(40, 100)
    ax2.set_title("Training loss")
    ax2.set_xlabel("Epoch")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("All models — same training settings")
    fig.tight_layout()
    out_png = os.path.join(cfg.output_dir, "comparison_curves.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    print(table_md)
    print(f"Saved: {out_md}, {out_png}")


if __name__ == "__main__":
    main()
