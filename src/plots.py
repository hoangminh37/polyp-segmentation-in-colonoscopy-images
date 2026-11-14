"""Learning curves (loss and validation Dice) from runs' log.csv.

    python -m src.plots runs/<run>                                  # -> runs/<run>/curves.png
    python -m src.plots runs/<a> runs/<b> runs/<c> --name models    # -> reports/models/curves.png

Training already saves curves.png into each run directory.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def output_path(runs, name, filename, reports_dir="reports"):
    """Where a figure belongs: inside the run for one run, reports/<name>/ for a comparison.

    A comparison folder also gets runs.txt listing exactly which runs it was built from.
    """
    if len(runs) == 1 and not name:
        return Path(runs[0]) / filename
    if not name:
        raise SystemExit("Comparing several runs: pass --name <comparison> (saved to reports/<comparison>/)")
    out_dir = Path(reports_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.txt").write_text("\n".join(Path(r).name for r in runs) + "\n")
    return out_dir / filename


def plot_curves(runs, out):
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        log = pd.read_csv(Path(run) / "log.csv")
        label = Path(run).name
        line, = ax_loss.plot(log["epoch"], log["train_loss"], linestyle="--", label=f"{label} (train)")
        ax_loss.plot(log["epoch"], log["val_loss"], color=line.get_color(), label=f"{label} (val)")
        ax_dice.plot(log["epoch"], log["val_dice"], color=line.get_color(), label=label)
        best = log["val_dice"].idxmax()
        ax_dice.scatter(log["epoch"][best], log["val_dice"][best], color=line.get_color(), zorder=3)
    ax_loss.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    ax_dice.set(xlabel="Epoch", ylabel="Dice", title="Validation Dice (dot = best checkpoint)")
    for ax in (ax_loss, ax_dice):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", help="comparison name; required when plotting several runs")
    args = ap.parse_args()
    print("Saved", plot_curves(args.runs, output_path(args.runs, args.name, "curves.png")))


if __name__ == "__main__":
    main()
