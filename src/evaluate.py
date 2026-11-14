"""Evaluate a trained run on the Kvasir test split or on an external dataset.

    python -m src.evaluate --run runs/unet_resnet34_bce_dice_adamw_cosine_s42
    python -m src.evaluate --run runs/<run> --data-dir data/raw/CVC-ClinicDB --name clinicdb

An external dataset is any folder with images/ and masks/ whose files share the same stem.
Results are written to <run>/metrics_<name>.json and <run>/per_image_<name>.csv.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader

from .data import PolypDataset, load_arrays, read_split
from .metrics import METRICS, batch_metrics
from .models import count_params, get_device, load_checkpoint


@torch.no_grad()
def evaluate_loader(model, loader, device, loss_fn=None, threshold=0.5):
    """Return {"mean": {metric: float}, "per_image": {metric: [float]}}."""
    model.eval()
    per_image = {m: [] for m in METRICS}
    loss_sum, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).float()
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * len(x)
        seen += len(x)
        for name, values in batch_metrics(logits, y, threshold).items():
            per_image[name] += values.cpu().tolist()
    mean = {m: sum(v) / len(v) for m, v in per_image.items()}
    if loss_fn is not None:
        mean["loss"] = loss_sum / seen
    return {"mean": mean, "per_image": per_image}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_latency(model, img_size, device, warmup=10, runs=50):
    """Mean milliseconds per image at batch size 1."""
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    _sync(device)
    return (time.perf_counter() - start) / runs * 1000


def evaluate_run(run_dir, data_dir=None, name=None, device=None, threshold=0.5):
    run_dir = Path(run_dir)
    device = device or get_device()
    model, cfg = load_checkpoint(run_dir / "best.pt", device)

    if data_dir is None:  # in-distribution test split
        names, images, masks = load_arrays(cfg["data_dir"], cfg["img_size"])
        subset = read_split(cfg["split_dir"], cfg["dataset_name"], "test")
        name = name or f"{cfg['dataset_name']}_test"
    else:  # whole external dataset
        names, images, masks = load_arrays(data_dir, cfg["img_size"])
        subset = None
        name = name or Path(data_dir).name.lower()
    dataset = PolypDataset(names, images, masks, subset)
    result = evaluate_loader(model, DataLoader(dataset, batch_size=16), device, threshold=threshold)

    summary = {
        "run": run_dir.name,
        "dataset": name,
        "num_images": len(dataset),
        "threshold": threshold,
        "mean": result["mean"],
        "params_millions": count_params(model) / 1e6,
        "latency_ms": {device.type: measure_latency(model, cfg["img_size"], device)},
        "cfg": cfg,
    }
    if device.type != "cpu":
        cpu_model = model.to("cpu")
        summary["latency_ms"]["cpu"] = measure_latency(cpu_model, cfg["img_size"], torch.device("cpu"), 3, 15)
    (run_dir / f"metrics_{name}.json").write_text(json.dumps(summary, indent=2))
    with open(run_dir / f"per_image_{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", *METRICS])
        for i, image_name in enumerate(dataset.names):
            writer.writerow([image_name, *(f"{result['per_image'][m][i]:.4f}" for m in METRICS)])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--data-dir", help="external dataset folder (images/ + masks/)")
    ap.add_argument("--name", help="dataset label used in output file names")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    summary = evaluate_run(args.run, args.data_dir, args.name, get_device(args.device), args.threshold)
    print(json.dumps({k: v for k, v in summary.items() if k != "cfg"}, indent=2))


if __name__ == "__main__":
    main()
