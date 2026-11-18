"""Run configuration: defaults, YAML loading and command-line overrides."""
import argparse
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "",  # run name; auto-generated when empty
    # --- model ---
    "model": "unet",  # custom_unet | unet | deeplabv3plus | segformer
    "encoder": "resnet34",  # smp encoder name (ignored by custom_unet)
    "encoder_weights": "imagenet",  # imagenet | none (train from scratch)
    # custom_unet only
    "base_channels": 32,  # channels of the first stage; doubled at each of the 4 downsamplings
    "skip": True,  # U-Net skip connections (False = plain encoder-decoder)
    "upsample": "transpose",  # transpose (learnable) | bilinear (fixed, followed by a 1x1 conv)
    "norm": "batch",  # batch | group | none
    "block": "plain",  # plain (two 3x3 convs) | residual (adds a ResNet-style shortcut)
    "dropout": 0.0,  # Dropout2d probability at the bottleneck
    # --- data ---
    "data_dir": "data/raw/Kvasir-SEG",
    "dataset_name": "kvasir",
    "split_dir": "data/splits",
    "img_size": 352,
    # --- optimisation ---
    "batch_size": 16,
    "epochs": 40,
    "optimizer": "adamw",  # adamw | adam | sgd
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "scheduler": "cosine",  # cosine | step | plateau | none
    "warmup_epochs": 0,  # linear warm-up before the schedule (not used with plateau)
    "loss": "bce_dice",  # see src/losses.py
    "pos_weight": 2.0,  # wbce: weight of polyp pixels (>1 penalises false negatives more)
    "tversky_alpha": 0.3,  # tversky: weight of false positives
    "tversky_beta": 0.7,  # tversky: weight of false negatives
    # --- misc ---
    "seed": 42,
    "device": "auto",
    "out_dir": "runs",
    "wandb": False,  # log to Weights & Biases
    "mlflow": False,  # log to MLflow (MLFLOW_TRACKING_URI, default mlruns/mlflow.db); see src/tracking.py
}


def with_defaults(cfg):
    """Fill keys missing from an older checkpoint's config."""
    return {**DEFAULTS, **cfg}


def parse_config(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            ap.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(f"--{key}", type=type(default), default=None)
    args = vars(ap.parse_args(argv))
    cfg = dict(DEFAULTS)
    if args["config"]:
        cfg.update(yaml.safe_load(Path(args["config"]).read_text()) or {})
    cfg.update({k: v for k, v in args.items() if k in DEFAULTS and v is not None})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("lr", "weight_decay", "dropout", "pos_weight", "tversky_alpha", "tversky_beta"):
        cfg[key] = float(cfg[key])
    if not cfg["name"]:
        cfg["name"] = "{model}_{encoder}_{loss}_{optimizer}_{scheduler}_s{seed}".format(**cfg)
    return cfg
