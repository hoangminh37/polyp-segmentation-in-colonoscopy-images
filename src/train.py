"""Train one segmentation model.

    python -m src.train --config configs/unet_resnet34.yaml
    python -m src.train --config configs/unet_resnet34.yaml --loss focal --seed 1
    python -m src.train --config configs/unet_resnet34.yaml --optimizer sgd --lr 0.01

Every key in DEFAULTS can be set in the YAML config and overridden on the command line.
Outputs go to runs/<run name>/: best.pt, log.csv, curves.png, config.yaml, metrics_kvasir_test.json
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import PolypDataset, load_arrays, read_split
from .evaluate import evaluate_loader, evaluate_run
from .losses import LOSSES
from .models import build_model, count_params, get_device
from .plots import plot_curves

DEFAULTS = {
    "name": "",  # run name; auto-generated when empty
    "model": "unet",  # unet | deeplabv3plus | segformer
    "encoder": "resnet34",
    "encoder_weights": "imagenet",
    "data_dir": "data/raw/Kvasir-SEG",
    "dataset_name": "kvasir",
    "split_dir": "data/splits",
    "img_size": 352,
    "batch_size": 16,
    "epochs": 60,
    "optimizer": "adamw",  # adamw | sgd
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "scheduler": "cosine",  # cosine | step | plateau | none
    "loss": "bce_dice",  # bce | dice | focal | bce_dice
    "seed": 42,
    "device": "auto",
    "out_dir": "runs",
    "wandb": False,
}


def parse_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            ap.add_argument(f"--{key}", action="store_true", default=None)
        else:
            ap.add_argument(f"--{key}", type=type(default), default=None)
    args = vars(ap.parse_args())
    cfg = dict(DEFAULTS)
    if args["config"]:
        cfg.update(yaml.safe_load(Path(args["config"]).read_text()) or {})
    cfg.update({k: v for k, v in args.items() if k in DEFAULTS and v is not None})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    cfg["lr"], cfg["weight_decay"] = float(cfg["lr"]), float(cfg["weight_decay"])
    if not cfg["name"]:
        cfg["name"] = "{model}_{encoder}_{loss}_{optimizer}_{scheduler}_s{seed}".format(**cfg)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(cfg, model):
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"], nesterov=True)
    raise ValueError(cfg["optimizer"])


def build_scheduler(cfg, optimizer):
    name, epochs = cfg["scheduler"], cfg["epochs"]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    if name == "none":
        return None
    raise ValueError(name)


def main():
    cfg = parse_config()
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = Path(cfg["out_dir"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    names, images, masks = load_arrays(cfg["data_dir"], cfg["img_size"])
    splits = {s: read_split(cfg["split_dir"], cfg["dataset_name"], s) for s in ("train", "val")}
    train_ds = PolypDataset(names, images, masks, splits["train"], train=True, seed=cfg["seed"])
    val_ds = PolypDataset(names, images, masks, splits["val"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    # drop_last: DeepLabV3+ pooling branch uses BatchNorm, which fails on a batch of 1.
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                              generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    model = build_model(cfg["model"], cfg["encoder"], cfg["encoder_weights"]).to(device)
    loss_fn = LOSSES[cfg["loss"]]
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"Run {cfg['name']} | device={device} | params={count_params(model) / 1e6:.1f}M | "
          f"train={len(train_ds)} val={len(val_ds)}")

    wandb_run = None
    if cfg["wandb"]:
        import wandb
        wandb_run = wandb.init(project="polyp-segmentation", name=cfg["name"], config=cfg)

    fields = ["epoch", "lr", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision",
              "val_recall", "seconds"]
    log_file = open(run_dir / "log.csv", "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fields)
    writer.writeheader()

    best_dice = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(x)
            seen += len(x)

        val = evaluate_loader(model, val_loader, device, loss_fn)["mean"]
        lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val["dice"])
        elif scheduler is not None:
            scheduler.step()

        row = {"epoch": epoch, "lr": lr, "train_loss": total / seen, "val_loss": val["loss"],
               "seconds": time.time() - start, **{f"val_{k}": val[k] for k in ("dice", "iou", "precision", "recall")}}
        writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        log_file.flush()
        if wandb_run:
            wandb_run.log(row)

        mark = ""
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_dice": best_dice},
                       run_dir / "best.pt")
            mark = " *"
        print(f"epoch {epoch:3d}/{cfg['epochs']} | train_loss {row['train_loss']:.4f} | "
              f"val_loss {val['loss']:.4f} | val_dice {val['dice']:.4f} | val_iou {val['iou']:.4f} | "
              f"{row['seconds']:.0f}s{mark}")

    log_file.close()
    print(f"Best val Dice: {best_dice:.4f}")
    plot_curves([run_dir], run_dir / "curves.png")
    test = evaluate_run(run_dir, device=device)
    print("Test:", {k: round(v, 4) for k, v in test["mean"].items()})
    if wandb_run:
        wandb_run.summary.update({f"test_{k}": v for k, v in test["mean"].items()})
        wandb_run.finish()


if __name__ == "__main__":
    main()
