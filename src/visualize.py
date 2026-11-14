"""Qualitative comparison grid: image | ground truth | prediction of each run.

    python -m src.visualize --runs runs/A --pick worst            # -> runs/A/qualitative_kvasir_test_worst.png
    python -m src.visualize --runs runs/A runs/B runs/C --name models
                                                                  # -> reports/models/qualitative_kvasir_test_random.png
With several runs, images are ranked (--pick worst/best) by the first run's Dice.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import PolypDataset, load_arrays, read_split
from .metrics import batch_metrics
from .models import get_device, load_checkpoint
from .plots import output_path


@torch.no_grad()
def predict(model, dataset, device):
    x = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y = torch.stack([dataset[i][1] for i in range(len(dataset))])
    logits = torch.cat([model(x[i:i + 16].to(device)).float().cpu() for i in range(0, len(x), 16)])
    return (torch.sigmoid(logits) > 0.5).squeeze(1).numpy(), batch_metrics(logits, y)["dice"].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-dir", help="external dataset folder; default is the Kvasir test split")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pick", choices=["random", "worst", "best"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="comparison name; required when comparing several runs")
    args = ap.parse_args()

    device = get_device()
    preds, dices, labels, dataset = [], [], [], None
    for run in args.runs:
        model, cfg = load_checkpoint(Path(run) / "best.pt", device)
        if dataset is None:
            names, images, masks = load_arrays(args.data_dir or cfg["data_dir"], cfg["img_size"])
            subset = None if args.data_dir else read_split(cfg["split_dir"], cfg["dataset_name"], "test")
            dataset = PolypDataset(names, images, masks, subset)
        pred, dice = predict(model, dataset, device)
        preds.append(pred)
        dices.append(dice)
        labels.append(f"{cfg['model']}\n{cfg['encoder']}")

    order = np.argsort(dices[0])
    if args.pick == "worst":
        idx = order[:args.n]
    elif args.pick == "best":
        idx = order[::-1][:args.n]
    else:
        idx = np.random.default_rng(args.seed).choice(len(dataset), size=min(args.n, len(dataset)), replace=False)

    cols = 2 + len(preds)
    fig, axes = plt.subplots(len(idx), cols, figsize=(2.4 * cols, 2.4 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        panels = [dataset.images[i], dataset.masks[i]] + [p[i] for p in preds]
        titles = ["Image", "Ground truth"] + labels
        for c, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[r, c]
            ax.imshow(panel, cmap=None if c == 0 else "gray", vmin=0, vmax=None if c == 0 else 1)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c >= 2:
                ax.text(4, 20, f"Dice {dices[c - 2][i]:.2f}", color="yellow", fontsize=8)
    fig.tight_layout()
    dataset_label = Path(args.data_dir).name.lower() if args.data_dir else f"{cfg['dataset_name']}_test"
    out = output_path(args.runs, args.name, f"qualitative_{dataset_label}_{args.pick}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
