"""Test-time augmentation (TTA) and model ensembling (L5), no extra training.

    python -m src.ensemble --runs runs/unet_r34_s42 runs/deeplabv3plus_r34_s42 runs/segformer_b2_s42

For each dataset (Kvasir test split, plus every external dataset present) it reports:
  - each model alone, without and with TTA (average of identity, h-flip, v-flip, both flips)
  - the ensemble (average of the models' probabilities), without and with TTA
Results go to reports/ensemble/results.csv.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch

from .data import PolypDataset, load_arrays, read_split
from .metrics import METRICS, batch_metrics
from .models import get_device, load_checkpoint

FLIPS = [[], [3], [2], [2, 3]]  # identity, horizontal, vertical, both


def _flipped_prob(model, x, dims):
    """Predict on a flipped copy and flip the prediction back."""
    if not dims:
        return torch.sigmoid(model(x).float())
    return torch.sigmoid(model(x.flip(dims)).float()).flip(dims)


@torch.no_grad()
def predict_probs(model, x_all, device, tta, batch_size=16):
    flips = FLIPS if tta else FLIPS[:1]
    out = []
    for i in range(0, len(x_all), batch_size):
        x = x_all[i:i + batch_size].to(device)
        out.append((sum(_flipped_prob(model, x, f) for f in flips) / len(flips)).cpu().half())
    return torch.cat(out)


def score(probs, y):
    logits = torch.logit(probs.float().clamp(1e-6, 1 - 1e-6))
    m = batch_metrics(logits, y)
    return {k: m[k].mean().item() for k in METRICS}


def datasets(cfg, externals):
    names, images, masks = load_arrays(cfg["data_dir"], cfg["img_size"])
    test = read_split(cfg["split_dir"], cfg["dataset_name"], "test")
    yield f"{cfg['dataset_name']}_test", PolypDataset(names, images, masks, test)
    for name, path in externals.items():
        if Path(path).exists():
            yield name, PolypDataset(*load_arrays(path, cfg["img_size"]))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiments import EXTERNAL_DATASETS

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="reports/ensemble/results.csv")
    args = ap.parse_args()

    device = get_device()
    models = [(Path(r).name, *load_checkpoint(Path(r) / "best.pt", device)) for r in args.runs]
    rows = []
    for ds_name, ds in datasets(models[0][2], EXTERNAL_DATASETS):
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.stack([ds[i][1] for i in range(len(ds))])
        ensemble = {False: 0, True: 0}
        for run, model, _ in models:
            for tta in (False, True):
                probs = predict_probs(model, x, device, tta)
                ensemble[tta] = ensemble[tta] + probs.float() / len(models)
                rows.append({"dataset": ds_name, "method": run, "tta": tta, **score(probs, y)})
        for tta in (False, True):
            rows.append({"dataset": ds_name, "method": "ensemble", "tta": tta, **score(ensemble[tta], y)})
        print(f"{ds_name}: done ({len(ds)} images)", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    (Path(args.out).parent / "runs.txt").write_text("\n".join(Path(r).name for r in args.runs) + "\n")
    df.to_csv(args.out, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
