"""Shared chart style and figure builders (static PNG for the report).

    python -m src.plots runs/<run>                                  # -> runs/<run>/curves.png
    python -m src.plots runs/<a> runs/<b> runs/<c> --name models    # -> reports/models/curves.png

Training already saves curves.png into each run directory. Group figures (dot plots, small
multiples, ...) are produced by scripts/make_report.py with the builders below.

Style: one fixed categorical colour per entity (never by rank), thin lines, hairline grid,
direct labels on line ends, gray for the baseline that variants are compared against.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE = INK["muted"]
DARK_BLUE, LIGHT_BLUE = "#256abf", "#86b6ef"  # two shades of one hue for paired values

plt.rcParams.update({
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
    "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8, "axes.labelcolor": INK["secondary"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelcolor": INK["secondary"],
    "ytick.labelcolor": INK["secondary"], "text.color": INK["primary"],
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


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


def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _direct_labels(ax, ends, min_gap_frac=0.06):
    """Write each series name at its line end, nudged apart so labels never collide."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    shifted = offset_copy(ax.transData, fig=ax.figure, x=6, units="points")
    placed = []
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.text(x, y_text, text, transform=shifted, va="center", fontsize=8, color=INK["secondary"],
                clip_on=False)


METRIC_TITLES = {"val_dice": "Validation Dice", "train_loss": "Training loss", "val_loss": "Validation loss",
                 "val_recall": "Validation recall", "val_precision": "Validation precision", "lr": "Learning rate"}


def plot_curves(runs, out):
    """Learning curves of one or a few runs: loss (train and val) and validation Dice."""
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, run in enumerate(runs):
        log = pd.read_csv(Path(run) / "log.csv")
        name = Path(run).name if len(runs) > 1 else ""
        c_train, c_val = (SERIES[0], SERIES[1]) if len(runs) == 1 else (SERIES[i], SERIES[i])
        ax_loss.plot(log["epoch"], log["train_loss"], color=c_train, alpha=0.55 if len(runs) > 1 else 1,
                     label=f"{name} train".strip())
        ax_loss.plot(log["epoch"], log["val_loss"], color=c_val, label=f"{name} validation".strip())
        ax_dice.plot(log["epoch"], log["val_dice"], color=SERIES[i], label=name or "validation Dice")
        best = log["val_dice"].idxmax()
        ax_dice.plot(log["epoch"][best], log["val_dice"][best], "o", ms=6, color=SERIES[i],
                     markeredgecolor=INK["surface"], markeredgewidth=1.5)
        ax_dice.annotate(f"best {log['val_dice'][best]:.3f} @ {log['epoch'][best]}",
                         (log["epoch"][best], log["val_dice"][best]), xytext=(0, -14),
                         textcoords="offset points", ha="center", fontsize=8, color=INK["secondary"])
    ax_loss.set(xlabel="Epoch", title="Loss")
    ax_dice.set(xlabel="Epoch", title="Validation Dice (dot = saved checkpoint)")
    for ax in (ax_loss, ax_dice):
        ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, out)


def curves_overlay(series, metrics, out, title=None):
    """series: [(label, log DataFrame, colour)]; one panel per metric, direct labels + legend."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics) + 1.2, 3.6), squeeze=False)
    for i, (ax, metric) in enumerate(zip(axes[0], metrics)):
        ends = []
        for label, log, colour in series:
            ax.plot(log["epoch"], log[metric], color=colour, label=label)
            ends.append((log["epoch"].iloc[-1], log[metric].iloc[-1], label))
        if metric == "lr":
            ax.set_yscale("log")
        if metric == "val_dice":
            lo = min(log[metric].quantile(0.1) for _, log, _ in series)
            ax.set_ylim(max(0, lo - 0.03), min(1, max(log[metric].max() for _, log, _ in series) + 0.02))
        ax.set(xlabel="Epoch", title=METRIC_TITLES.get(metric, metric))
        ax.margins(x=0.02)
        if i == 0:
            _direct_labels(ax, ends)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.08))
    if title:
        fig.suptitle(title, x=0.01, ha="left", y=1.15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def curves_small_multiples(baseline, variants, metric, out, title=None, ncols=3):
    """One panel per variant: the variant (blue) against the shared baseline (gray), same y-axis."""
    n = len(variants)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.5 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    base_label, base_log = baseline
    values = pd.concat([base_log[metric]] + [log[metric] for _, log in variants])
    for ax, (label, log) in zip(axes.flat, variants):
        ax.plot(base_log["epoch"], base_log[metric], color=BASELINE, label=base_label)
        ax.plot(log["epoch"], log[metric], color=SERIES[0], label="Variant")
        ax.set_title(label, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    if metric == "lr":
        axes[0][0].set_yscale("log")
    elif metric.startswith("val_"):
        axes[0][0].set_ylim(max(0, values.quantile(0.05) - 0.03), min(1, values.max() + 0.02))
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    for row in axes:
        row[0].set_ylabel(METRIC_TITLES.get(metric, metric))
    handles = [plt.Line2D([], [], color=BASELINE), plt.Line2D([], [], color=SERIES[0])]
    fig.legend(handles, [f"Baseline: {base_label}", "Variant (panel title)"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04))
    if title:
        fig.suptitle(title, x=0.01, ha="left", y=1.09, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def dot_plot(table, metrics, out, baseline=None, title=None):
    """Test metrics per variant: dot = mean, whisker = ±1 std over seeds (when > 1 seed).

    table: DataFrame indexed by label with columns <metric>_mean, <metric>_std, seeds.
    """
    labels = list(table.index)
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics) + 2.2, 0.38 * len(labels) + 1.2),
                             sharey=True, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        mean, std = table[f"{metric}_mean"].to_numpy(), table[f"{metric}_std"].fillna(0).to_numpy()
        colours = [BASELINE if lab == baseline else SERIES[0] for lab in labels]
        ax.errorbar(mean, y, xerr=std, fmt="none", ecolor=INK["axis"], elinewidth=1.2, capsize=0)
        ax.scatter(mean, y, s=42, c=colours, zorder=3, edgecolors=INK["surface"], linewidths=1.2)
        if baseline in labels:
            ax.axvline(mean[labels.index(baseline)], color=INK["axis"], linewidth=0.8, zorder=1)
        lo, hi = np.nanmin(mean - std), np.nanmax(mean + std)
        pad = max(0.01, (hi - lo) * 0.15)
        ax.set_xlim(max(0, lo - pad), min(1, hi + pad))
        ax.set_title(metric.capitalize() if metric != "iou" else "IoU")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y, labels)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def dumbbell(table, a_col, b_col, a_label, b_label, out, title=None):
    """Two values per row (e.g. in-domain vs external Dice), joined by a hairline."""
    labels = list(table.index)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(labels) + 1.3))
    a, b = table[a_col].to_numpy(), table[b_col].to_numpy()
    ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color=INK["axis"], linewidth=1.2)
    ax.scatter(a, y, s=48, color=DARK_BLUE, zorder=3, label=a_label, edgecolors=INK["surface"], linewidths=1.2)
    ax.scatter(b, y, s=48, color=LIGHT_BLUE, zorder=3, label=b_label, edgecolors=INK["surface"], linewidths=1.2)
    for yi, ai, bi in zip(y, a, b):
        ax.annotate(f"−{ai - bi:.3f}" if ai >= bi else f"+{bi - ai:.3f}", ((ai + bi) / 2, yi),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=INK["secondary"])
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Dice")
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def lines_by_category(series, categories, out, ylabel, title=None):
    """series: [(label, [value per category], colour)] over an ordered categorical x-axis."""
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(categories))
    ends = []
    for label, values, colour in series:
        ax.plot(x, values, "-o", color=colour, ms=5, label=label, markeredgecolor=INK["surface"])
        ends.append((x[-1], values[-1], label))
    ax.set_xticks(x, categories)
    ax.set_xlim(-0.2, len(categories) - 0.2)
    ax.set_ylabel(ylabel)
    _direct_labels(ax, ends)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=min(3, len(series)))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def labelled_scatter(points, out, xlabel, ylabel, title=None, logx=False):
    """points: [(label, x, y)]; one hue, every point named directly."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, x, y in points:
        ax.scatter(x, y, s=48, color=SERIES[0], edgecolors=INK["surface"], linewidths=1.2, zorder=3)
        ax.annotate(label, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK["secondary"])
    if logx:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.margins(x=0.25, y=0.2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", help="comparison name; required when plotting several runs")
    args = ap.parse_args()
    print("Saved", plot_curves(args.runs, output_path(args.runs, args.name, "curves.png")))


if __name__ == "__main__":
    main()
