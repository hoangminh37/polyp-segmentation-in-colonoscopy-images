"""Experiment grid: every run of the project, grouped for reporting, plus a resumable runner.

    python scripts/experiments.py --list                         # show groups and run status
    python scripts/experiments.py --group main --seeds 42        # train what is missing
    python scripts/experiments.py --group loss --group optim     # several groups

A run is skipped when runs/<name>/metrics_kvasir_test.json exists, so the runner can be
stopped and restarted. After training, each run is also evaluated on every external dataset
in EXTERNAL_DATASETS that is present on disk.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 1, 2]
EXTERNAL_DATASETS = {"clinicdb": "data/raw/CVC-ClinicDB"}  # evaluation only, never trained on

CUSTOM = "configs/custom_unet.yaml"
UNET = "configs/unet_resnet34.yaml"
DEEPLAB = "configs/deeplabv3plus_resnet34.yaml"
SEGFORMER = "configs/segformer_mitb2.yaml"

# prefix -> (label, config, extra args)
MAIN = {
    "custom_unet": ("Custom U-Net (scratch)", CUSTOM, []),
    "unet_r34_scratch": ("U-Net R34 (scratch)", UNET, ["--encoder_weights", "none"]),
    "unet_r34": ("U-Net R34 (ImageNet)", UNET, []),
    "deeplabv3plus_r34": ("DeepLabV3+ R34 (ImageNet)", DEEPLAB, []),
    "segformer_b2": ("SegFormer B2 (ImageNet)", SEGFORMER, []),
}

# Ablations: one seed, one factor changed against the group's baseline run.
ARCH = {  # on the from-scratch custom U-Net
    "custom_unet": ("Baseline (skip, transposed conv, BatchNorm)", CUSTOM, []),
    "custom_unet_noskip": ("No skip connections", CUSTOM, ["--no-skip"]),
    "custom_unet_bilinear": ("Bilinear upsampling", CUSTOM, ["--upsample", "bilinear"]),
    "custom_unet_groupnorm": ("GroupNorm", CUSTOM, ["--norm", "group"]),
    "custom_unet_nonorm": ("No normalisation", CUSTOM, ["--norm", "none"]),
    "custom_unet_residual": ("Residual blocks", CUSTOM, ["--block", "residual"]),
    "custom_unet_dropout": ("Dropout 0.3 at bottleneck", CUSTOM, ["--dropout", "0.3"]),
}
LOSS = {  # on U-Net R34 (ImageNet); baseline uses BCE + Dice
    "unet_r34": ("BCE + Dice (baseline)", UNET, []),
    "unet_r34_loss-bce": ("BCE", UNET, ["--loss", "bce"]),
    "unet_r34_loss-wbce": ("Weighted BCE (β = 2)", UNET, ["--loss", "wbce"]),
    "unet_r34_loss-focal": ("Focal", UNET, ["--loss", "focal"]),
    "unet_r34_loss-dice": ("Dice", UNET, ["--loss", "dice"]),
    "unet_r34_loss-tversky": ("Tversky (α 0.3, β 0.7)", UNET, ["--loss", "tversky"]),
    "unet_r34_loss-dicefocal": ("Dice + Focal", UNET, ["--loss", "dice_focal"]),
}
OPTIM = {  # on U-Net R34 (ImageNet); baseline is AdamW + cosine. Each optimizer uses its usual weight decay.
    "unet_r34": ("AdamW, cosine (baseline)", UNET, []),
    "unet_r34_opt-adam": ("Adam (L2 1e-4), cosine", UNET, ["--optimizer", "adam", "--weight_decay", "1e-4"]),
    "unet_r34_opt-sgd": ("SGD momentum lr 0.01, cosine", UNET,
                         ["--optimizer", "sgd", "--lr", "0.01", "--weight_decay", "1e-4"]),
    "unet_r34_opt-sgd-warmup": ("SGD + 5 warm-up epochs, cosine", UNET,
                                ["--optimizer", "sgd", "--lr", "0.01", "--weight_decay", "1e-4",
                                 "--warmup_epochs", "5"]),
    "unet_r34_sched-step": ("AdamW, step decay", UNET, ["--scheduler", "step"]),
    "unet_r34_sched-plateau": ("AdamW, reduce on plateau", UNET, ["--scheduler", "plateau"]),
    "unet_r34_sched-constant": ("AdamW, constant LR", UNET, ["--scheduler", "none"]),
}
GROUPS = {"main": MAIN, "arch": ARCH, "loss": LOSS, "optim": OPTIM}
MULTI_SEED = {"main"}  # ablations use seed 42 only


def runs_of(group, seeds=None):
    """[(run name, label, prefix, config, args, seed)] for a group."""
    seeds = seeds or (SEEDS if group in MULTI_SEED else [42])
    return [(f"{prefix}_s{seed}", label, prefix, config, args, seed)
            for seed in seeds for prefix, (label, config, args) in GROUPS[group].items()]


def is_done(name, dataset="kvasir_test"):
    return (ROOT / "runs" / name / f"metrics_{dataset}.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", choices=list(GROUPS), help="default: all groups")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the seeds of multi-seed groups")
    ap.add_argument("--only", nargs="+", help="run only these prefixes (e.g. unet_r34 segformer_b2)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    queue, seen = [], set()
    for group in args.group or list(GROUPS):
        seeds = args.seeds if group in MULTI_SEED else None
        for run in runs_of(group, seeds):
            if run[0] not in seen and (not args.only or run[2] in args.only):
                seen.add(run[0])
                queue.append(run)

    if args.list:
        for name, label, *_ in queue:
            print(f"{'done' if is_done(name) else 'todo':4s}  {name:32s} {label}")
        return

    py = sys.executable
    for i, (name, label, _, config, extra, seed) in enumerate(queue, 1):
        print(f"\n=== [{i}/{len(queue)}] {name}: {label}", flush=True)
        if is_done(name):
            print("already trained, skipping", flush=True)
        else:
            cmd = [py, "-m", "src.train", "--config", config, "--name", name, "--seed", str(seed), *extra]
            subprocess.run(cmd, cwd=ROOT, check=True)
        for ds_name, ds_dir in EXTERNAL_DATASETS.items():
            if (ROOT / ds_dir).exists() and not is_done(name, ds_name):
                subprocess.run([py, "-m", "src.evaluate", "--run", f"runs/{name}", "--data-dir", ds_dir,
                                "--name", ds_name, "--quiet"], cwd=ROOT, check=True)
                print(f"evaluated on {ds_name}", flush=True)
    print("\nAll queued runs finished.", flush=True)


if __name__ == "__main__":
    main()
