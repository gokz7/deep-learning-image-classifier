"""Training script.
Usage:  python -m src.train --model baseline            (full run, epochs from config)
        python -m src.train --model baseline --epochs 2 (quick test run)
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.config import Config
from src.data import get_dataloaders
from src.models import get_model, count_params
from src.utils import set_seed, save_checkpoint, plot_curves

try:
    import mlflow
except ImportError:
    mlflow = None


# ---------- One epoch of training ----------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()   # backward pass (scaled for mixed precision)
        scaler.step(optimizer)          # update weights
        scaler.update()
        scheduler.step()                # OneCycle: update LR every batch

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# ---------- Measure on validation/test (no training) ----------
@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# ---------- MLflow helpers (training never fails because of logging) ----------
def mlflow_start(cfg, model_name, n_params):
    if mlflow is None:
        print("[MLflow] not installed, skipping tracking")
        return False
    try:
        mlflow_dir = os.path.abspath(cfg.mlflow_dir)
        os.makedirs(mlflow_dir, exist_ok=True)
        # Runs are stored in a SQLite database file; plots/CSVs go to mlruns/artifacts
        mlflow.set_tracking_uri("sqlite:///" + os.path.join(mlflow_dir, "mlflow.db"))
        if mlflow.get_experiment_by_name("cifar10") is None:
            artifact_dir = Path(mlflow_dir, "artifacts").as_uri()
            mlflow.create_experiment("cifar10", artifact_location=artifact_dir)
        mlflow.set_experiment("cifar10")
        mlflow.start_run(run_name=f"{model_name}_{cfg.epochs}ep")
        mlflow.log_params({**cfg.to_dict(), "model": model_name, "trainable_params": n_params})
        return True
    except Exception as e:
        print(f"[MLflow] disabled: {e}")
        return False


def mlflow_safe(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"[MLflow] logging error (ignored): {e}")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    set_seed(cfg.seed)
    device = cfg.device
    use_amp = cfg.use_amp and device == "cuda"

    # Data: train + validation only. Test set is NOT used during training.
    train_loader, val_loader, _ = get_dataloaders(cfg)

    model = get_model(args.model, cfg).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
        nesterov=cfg.nesterov, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(train_loader),
    )
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    # Output paths
    out_dir = os.path.join(cfg.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoint_dir, f"best_{args.model}.pt")
    last_path = os.path.join(cfg.checkpoint_dir, f"last_{args.model}.pt")

    use_mlflow = mlflow_start(cfg, args.model, n_params)

    print(f"Model: {args.model} | params: {n_params:,} | device: {device} | AMP: {use_amp} | epochs: {cfg.epochs}", flush=True)

    history = []
    best_val_acc, best_epoch = 0.0, 0
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, use_amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        epoch_time = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now, "epoch_time_sec": epoch_time,
        })

        # Checkpoints
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc, best_epoch = val_acc, epoch
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_val_acc, cfg)
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_val_acc, cfg)

        # Save history every epoch (safe if Colab disconnects)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

        if use_mlflow:
            mlflow_safe(mlflow.log_metrics, {
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
            }, step=epoch)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | lr {lr_now:.4f} | "
              f"train loss {train_loss:.4f} acc {train_acc*100:5.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:5.2f}% | {epoch_time:5.1f}s"
              + ("  <- best" if is_best else ""), flush=True)

    total_time = time.time() - start_time

    # Curves + summary
    curves_path = os.path.join(out_dir, "curves.png")
    plot_curves(history, curves_path, title=f"{args.model} — training curves")
    summary = {
        "model": args.model, "trainable_params": n_params, "epochs": cfg.epochs,
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "total_train_time_sec": round(total_time, 1),
        "avg_epoch_time_sec": round(total_time / cfg.epochs, 1),
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_mlflow:
        mlflow_safe(mlflow.log_metrics, {"best_val_acc": best_val_acc, "best_epoch": best_epoch})
        for fname in ["curves.png", "history.csv", "train_summary.json"]:
            mlflow_safe(mlflow.log_artifact, os.path.join(out_dir, fname))
        mlflow_safe(mlflow.end_run)

    print(f"\nDone in {total_time/60:.1f} min | best val acc {best_val_acc*100:.2f}% at epoch {best_epoch}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
