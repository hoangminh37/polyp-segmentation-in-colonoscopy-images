"""Collect runs/*/metrics_*.json into one table (mean ± std over seeds).

    python scripts/summarize.py     # prints Markdown; writes reports/all_runs.csv (one row per run
                                    # and dataset, with the run name) and reports/summary.md
"""
import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["model", "encoder", "loss", "optimizer", "scheduler"]
METRICS = ["dice", "iou", "precision", "recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.runs_dir).glob("*/metrics_*.json")):
        m = json.loads(path.read_text())
        rows.append({
            "run": m["run"],
            "dataset": m["dataset"],
            **{k: m["cfg"][k] for k in KEYS},
            "seed": m["cfg"]["seed"],
            **{k: m["mean"][k] for k in METRICS},
            "params_M": m["params_millions"],
            "cpu_ms": m["latency_ms"].get("cpu"),
        })
    if not rows:
        raise SystemExit(f"No metrics_*.json found under {args.runs_dir}")
    df = pd.DataFrame(rows)

    grouped = df.groupby(["dataset", *KEYS])
    table = grouped[["params_M", "cpu_ms"]].mean().round(1)
    table.insert(0, "seeds", grouped["seed"].nunique())
    for metric in METRICS:
        mean, std = grouped[metric].mean(), grouped[metric].std().fillna(0)
        table[metric] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(mean, std)]
    table = table.reset_index()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_runs.csv", index=False)
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for _, row in table.iterrows()]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}/all_runs.csv and {out_dir}/summary.md")


if __name__ == "__main__":
    main()
