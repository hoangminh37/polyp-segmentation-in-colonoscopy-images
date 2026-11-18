"""Build every comparison table and figure from the finished runs.

    python scripts/make_report.py

Reads runs/*/metrics_*.json, log.csv and per_image_*.csv (only finished runs are used, so it can
be re-run while experiments are still going) and writes:

    reports/all_runs.csv            one row per run and dataset
    reports/summary.md              all tables, with links to the figures
    reports/<group>/...             figures of each experiment group (main, arch, loss, optim)
    reports/ensemble/...            figure of reports/ensemble/results.csv when present
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import EXTERNAL_DATASETS, GROUPS, MAIN  # noqa: E402
from src import plots  # noqa: E402
from src.data import load_arrays, read_split  # noqa: E402

RUNS, REPORTS = ROOT / "runs", ROOT / "reports"  # overridable with --runs-dir / --reports-dir
METRICS = ["dice", "iou", "precision", "recall"]
IN_DOMAIN = "kvasir_test"
MAIN_COLOURS = {prefix: plots.SERIES[i] for i, prefix in enumerate(MAIN)}  # fixed per model
SIZE_BINS = [(0, 0.05, "small (< 5%)"), (0.05, 0.20, "medium (5–20%)"), (0.20, 1.01, "large (> 20%)")]
GROUP_TITLES = {
    "main": "Main comparison: from-scratch baseline → transfer learning → advanced architectures",
    "arch": "Architecture ablation on the from-scratch custom U-Net (seed 42)",
    "loss": "Loss ablation on U-Net R34 (ImageNet), seed 42",
    "optim": "Optimizer and LR-schedule ablation on U-Net R34 (ImageNet), seed 42",
}


def split_name(run):
    m = re.match(r"(.+)_s(\d+)$", run)
    return (m.group(1), int(m.group(2))) if m else (run, None)


def collect():
    rows = []
    for path in sorted(RUNS.glob("*/metrics_*.json")):
        m = json.loads(path.read_text())
        prefix, seed = split_name(m["run"])
        rows.append({"run": m["run"], "prefix": prefix, "seed": seed, "dataset": m["dataset"],
                     **{k: m["mean"][k] for k in METRICS}, "params_M": m["params_millions"],
                     "cpu_ms": m["latency_ms"].get("cpu"), "gpu_ms": m["latency_ms"].get("mps", m["latency_ms"].get("cuda"))})
    return pd.DataFrame(rows)


def aggregate(df, prefixes, labels, dataset):
    sub = df[(df.dataset == dataset) & df.prefix.isin(prefixes)]
    if sub.empty:
        return None
    g = sub.groupby("prefix")
    table = pd.DataFrame({"seeds": g.seed.nunique(), "params_M": g.params_M.mean(), "cpu_ms": g.cpu_ms.mean()})
    for k in METRICS:
        table[f"{k}_mean"], table[f"{k}_std"] = g[k].mean(), g[k].std()
    table = table.loc[[p for p in prefixes if p in table.index]]
    table.index = [labels[p] for p in table.index]
    return table


def fmt(mean, std, seeds):
    return f"{mean:.4f} ± {std:.4f}" if seeds > 1 and not np.isnan(std) else f"{mean:.4f}"


def markdown_table(table, external=None):
    cols = ["Variant", "Seeds", "Params (M)", "CPU ms", "Dice", "IoU", "Precision", "Recall"]
    if external is not None:
        cols.append("Dice on ClinicDB")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for label, r in table.iterrows():
        cells = [label, str(int(r.seeds)), f"{r.params_M:.1f}", f"{r.cpu_ms:.0f}"]
        cells += [fmt(r[f"{k}_mean"], r[f"{k}_std"], r.seeds) for k in METRICS]
        if external is not None:
            cells.append(fmt(external.loc[label, "dice_mean"], external.loc[label, "dice_std"],
                             external.loc[label, "seeds"]) if label in external.index else "–")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_log(run):
    path = RUNS / run / "log.csv"
    return pd.read_csv(path) if path.exists() else None


def size_analysis(df_runs, dataset, data_dir, split):
    """Mean Dice per polyp-size bin for each run (size = mask area / image area)."""
    names, _, masks = load_arrays(data_dir, 352)
    area = dict(zip(names, masks.reshape(len(masks), -1).mean(1)))
    if split:
        area = {n: area[n] for n in split}
    counts = [sum(lo <= a < hi for a in area.values()) for lo, hi, _ in SIZE_BINS]
    series = []
    for prefix, run in df_runs:
        per = pd.read_csv(RUNS / run / f"per_image_{dataset}.csv", dtype={"image": str})
        per["area"] = per.image.map(area)
        series.append((prefix, [per[(per.area >= lo) & (per.area < hi)].dice.mean() for lo, hi, _ in SIZE_BINS]))
    categories = [f"{name}\nn = {n}" for (_, _, name), n in zip(SIZE_BINS, counts)]
    return series, categories


def report_group(group, df, md, figures):
    spec = GROUPS[group]
    labels = {p: (MAIN[p][0] if group == "main" else spec[p][0]) for p in spec}
    table = aggregate(df, list(spec), labels, IN_DOMAIN)
    if table is None or len(table) < 2:
        md.append(f"## {GROUP_TITLES[group]}\n\nNot enough finished runs yet "
                  f"({0 if table is None else len(table)} of {len(spec)}).\n")
        return
    out = REPORTS / group
    out.mkdir(parents=True, exist_ok=True)
    external = aggregate(df, list(spec), labels, "clinicdb")
    md.append(f"## {GROUP_TITLES[group]}\n")
    md.append(f"{len(table)} of {len(spec)} variants finished. Kvasir-SEG test split (100 images); "
              "mean ± std over seeds where more than one seed exists.\n")
    md.append(markdown_table(table, external) + "\n")

    baseline_prefix = list(spec)[0]
    baseline_label = labels[baseline_prefix] if group != "main" else None
    plots.dot_plot(table, METRICS, out / "test_metrics.png", baseline=baseline_label,
                   title=f"Test metrics on Kvasir-SEG ({'dot = mean, whisker = ±1 std' if table.seeds.max() > 1 else 'seed 42'})")
    figures.append((group, "test_metrics.png", "Test metrics on Kvasir-SEG"))
    if external is not None and len(external) >= 2:
        plots.dot_plot(external, METRICS, out / "test_metrics_clinicdb.png", baseline=baseline_label,
                       title="Test metrics on CVC-ClinicDB (external, never trained on)")
        figures.append((group, "test_metrics_clinicdb.png", "Test metrics on CVC-ClinicDB"))

    logs = {p: read_log(f"{p}_s42") for p in spec}
    logs = {p: log for p, log in logs.items() if log is not None}
    if group == "main":
        series = [(labels[p], log, MAIN_COLOURS[p]) for p, log in logs.items()]
        plots.curves_overlay(series, ["val_dice", "train_loss", "val_loss"], out / "curves.png",
                             title="Learning curves (seed 42)")
        figures.append((group, "curves.png", "Learning curves"))
    elif baseline_prefix in logs and len(logs) >= 2:
        variants = [(labels[p], log) for p, log in logs.items() if p != baseline_prefix]
        base = (labels[baseline_prefix], logs[baseline_prefix])
        plots.curves_small_multiples(base, variants, "val_dice", out / "curves_val_dice.png",
                                     title="Validation Dice per epoch, each variant against the baseline")
        figures.append((group, "curves_val_dice.png", "Validation Dice curves"))
        second = {"arch": "train_loss", "loss": "val_recall", "optim": "lr"}[group]
        plots.curves_small_multiples(base, variants, second, out / f"curves_{second}.png",
                                     title=f"{plots.METRIC_TITLES[second]} per epoch")
        figures.append((group, f"curves_{second}.png", plots.METRIC_TITLES[second]))

    if group == "main":
        report_main_extras(df, labels, out, md, figures)


def report_main_extras(df, labels, out, md, figures):
    done = [p for p in MAIN if (RUNS / f"{p}_s42" / f"metrics_{IN_DOMAIN}.json").exists()]
    runs42 = [(p, f"{p}_s42") for p in done]

    # in-domain vs external
    ext = df[(df.dataset == "clinicdb") & (df.seed == 42) & df.prefix.isin(done)].set_index("prefix")
    ind = df[(df.dataset == IN_DOMAIN) & (df.seed == 42) & df.prefix.isin(done)].set_index("prefix")
    common = [p for p in done if p in ext.index]
    if common:
        t = pd.DataFrame({"kvasir": ind.loc[common, "dice"], "clinicdb": ext.loc[common, "dice"]})
        t.index = [labels[p] for p in common]
        plots.dumbbell(t, "kvasir", "clinicdb", "Kvasir-SEG test (same hospital as training)",
                       "CVC-ClinicDB (other hospital, never seen)", out / "generalisation.png",
                       title="Dice drop on an external dataset (seed 42)")
        figures.append(("main", "generalisation.png", "In-domain vs external Dice"))

    # polyp size
    cfg_split = read_split(ROOT / "data/splits", "kvasir", "test")
    for dataset, data_dir, split in [(IN_DOMAIN, ROOT / "data/raw/Kvasir-SEG", cfg_split),
                                     *[(n, ROOT / d, None) for n, d in EXTERNAL_DATASETS.items()]]:
        runs = [(p, r) for p, r in runs42 if (RUNS / r / f"per_image_{dataset}.csv").exists()]
        if not runs:
            continue
        series, categories = size_analysis(runs, dataset, data_dir, split)
        plots.lines_by_category([(labels[p], v, MAIN_COLOURS[p]) for p, v in series], categories,
                                out / f"dice_by_size_{dataset}.png", "Mean Dice",
                                title=f"Dice by polyp size ({dataset}, seed 42)")
        figures.append(("main", f"dice_by_size_{dataset}.png", f"Dice by polyp size ({dataset})"))

    # efficiency
    points = [(f"{labels[p]} · {ind.loc[p, 'params_M']:.1f}M", ind.loc[p, "cpu_ms"], ind.loc[p, "dice"])
              for p in done]
    plots.labelled_scatter(points, out / "efficiency.png", "CPU inference time per image (ms)",
                           "Test Dice (Kvasir-SEG)", title="Accuracy vs CPU inference time (seed 42)")
    figures.append(("main", "efficiency.png", "Accuracy vs inference time"))

    # qualitative grids
    run_dirs = [str(RUNS / r) for _, r in runs42]
    for args, fname in [(["--pick", "random"], "qualitative_kvasir_test_random.png"),
                        (["--pick", "worst"], "qualitative_kvasir_test_worst.png")] + \
            [(["--pick", "random", "--data-dir", str(ROOT / d)], f"qualitative_{Path(d).name.lower()}_random.png")
             for d in EXTERNAL_DATASETS.values() if (ROOT / d).exists()]:
        subprocess.run([sys.executable, "-m", "src.visualize", "--runs", *run_dirs, "--name", "main",
                        "--reports-dir", str(REPORTS), "--n", "6", *args], cwd=ROOT, check=True,
                       capture_output=True)
        figures.append(("main", fname, "Qualitative examples: " + fname.removeprefix("qualitative_")
                        .removesuffix(".png").replace("_", " ")))


def report_ensemble(md, figures):
    path = REPORTS / "ensemble" / "results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    names = {r: MAIN[split_name(r)[0]][0] if split_name(r)[0] in MAIN else r for r in df.method.unique()}
    names["ensemble"] = "Ensemble (mean probability)"
    md.append("## Test-time augmentation and ensembling (no extra training)\n")
    md.append("Runs: " + ", ".join(f"`{r}`" for r in df.method.unique() if r != "ensemble") + ".\n")
    for dataset, sub in df.groupby("dataset", sort=False):
        md.append(f"**{dataset}**\n")
        md.append("| Method | Dice | Dice with TTA | IoU | IoU with TTA |\n|---|---|---|---|---|")
        for method, rows in sub.groupby("method", sort=False):
            a, b = rows[~rows.tta].iloc[0], rows[rows.tta].iloc[0]
            md.append(f"| {names[method]} | {a.dice:.4f} | {b.dice:.4f} | {a.iou:.4f} | {b.iou:.4f} |")
        md.append("")
        t = sub.pivot_table(index="method", columns="tta", values="dice", sort=False)
        t.index = [names[m] for m in t.index]
        plots.dumbbell(t, False, True, "Single pass", "With TTA (4 flips)",
                       REPORTS / "ensemble" / f"tta_{dataset}.png", title=f"Effect of TTA and ensembling ({dataset})")
        figures.append(("ensemble", f"tta_{dataset}.png", f"TTA and ensemble ({dataset})"))


def main():
    global RUNS, REPORTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--reports-dir", default=str(REPORTS))
    args = ap.parse_args()
    RUNS, REPORTS = Path(args.runs_dir).resolve(), Path(args.reports_dir).resolve()
    df = collect()
    if df.empty:
        raise SystemExit("No finished runs under runs/")
    REPORTS.mkdir(exist_ok=True)
    df.sort_values(["dataset", "prefix", "seed"]).to_csv(REPORTS / "all_runs.csv", index=False)
    md = ["# Experiment results\n",
          "Generated by `python scripts/make_report.py` from the runs in `runs/`. Metrics are computed per "
          "image at 352×352 and averaged over images (threshold 0.5). CPU ms = inference time per image "
          "on the training laptop's CPU (Apple M4), batch size 1.\n"]
    figures = []
    for group in GROUPS:
        report_group(group, df, md, figures)
    report_ensemble(md, figures)
    md.append("## Figures\n")
    for folder, fname, caption in figures:
        md.append(f"- [{caption}]({folder}/{fname})")
    (REPORTS / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {REPORTS / 'summary.md'} and {len(figures)} figures")


if __name__ == "__main__":
    main()
