"""Export a trained model to ONNX and check it matches PyTorch.
Usage: python -m src.export_onnx --model deep_residual
"""
import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.config import Config
from src.data import get_dataloaders
from src.models import get_model
from src.utils import set_seed, load_checkpoint


def export_to_onnx(model, path):
    """Export with a flexible batch size. Tries the classic exporter first, then the newer one."""
    dummy = torch.randn(2, 3, 32, 32)
    try:
        torch.onnx.export(
            model, (dummy,), path,
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        return "torchscript exporter"
    except Exception as e:
        print(f"Classic exporter not available ({type(e).__name__}); using dynamo exporter")
        torch.onnx.export(
            model, (dummy,), path,
            input_names=["image"], output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
            dynamo=True,
        )
        return "dynamo exporter"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deep_residual")
    parser.add_argument("--n_images", type=int, default=1000)
    args = parser.parse_args()

    cfg = Config()
    cfg.device = "cpu"          # export and compare on CPU
    set_seed(cfg.seed)

    # 1. Load best checkpoint
    model = get_model(args.model, cfg)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"best_{args.model}.pt")
    load_checkpoint(ckpt_path, model, "cpu")
    model.eval()

    # 2. Export + validate the file
    onnx_path = os.path.join(cfg.checkpoint_dir, f"{args.model}.onnx")
    method = export_to_onnx(model, onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"Exported {onnx_path} ({size_mb:.1f} MB) using {method}; ONNX check passed")

    # 3. Compare PyTorch vs ONNX Runtime on test images
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    _, _, test_loader = get_dataloaders(cfg)

    max_diff, agree, correct_pt, correct_ox, n = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            pt_logits = model(images).numpy()
            ox_logits = session.run(None, {input_name: images.numpy()})[0]
            max_diff = max(max_diff, float(np.abs(pt_logits - ox_logits).max()))
            pt_pred, ox_pred = pt_logits.argmax(1), ox_logits.argmax(1)
            agree += int((pt_pred == ox_pred).sum())
            correct_pt += int((pt_pred == labels.numpy()).sum())
            correct_ox += int((ox_pred == labels.numpy()).sum())
            n += len(labels)
            if n >= args.n_images:
                break

    # 4. Speed: one image at a time on CPU
    one = np.random.randn(1, 3, 32, 32).astype(np.float32)
    for _ in range(10):
        session.run(None, {input_name: one})            # warm-up
    t0 = time.time()
    for _ in range(100):
        session.run(None, {input_name: one})
    ms_per_image = (time.time() - t0) / 100 * 1000

    result = {
        "model": args.model,
        "onnx_file": onnx_path,
        "onnx_size_mb": round(size_mb, 2),
        "export_method": method,
        "images_compared": n,
        "max_abs_logit_diff": max_diff,
        "prediction_agreement": agree / n,
        "pytorch_acc_on_subset": correct_pt / n,
        "onnx_acc_on_subset": correct_ox / n,
        "onnx_cpu_ms_per_image": round(ms_per_image, 2),
    }
    with open(os.path.join(cfg.output_dir, "onnx_check.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCompared {n} test images:")
    print(f"  Max difference in output scores: {max_diff:.2e}")
    print(f"  Same predicted class:            {agree / n * 100:.2f}%")
    print(f"  PyTorch accuracy on these:       {correct_pt / n * 100:.2f}%")
    print(f"  ONNX accuracy on these:          {correct_ox / n * 100:.2f}%")
    print(f"  ONNX Runtime CPU speed:          {ms_per_image:.2f} ms per image")
    print(f"\nSaved: {cfg.output_dir}/onnx_check.json")


if __name__ == "__main__":
    main()
