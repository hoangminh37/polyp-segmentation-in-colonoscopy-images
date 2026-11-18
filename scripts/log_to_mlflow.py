"""Import finished runs (and the current model release) into MLflow.

    python scripts/log_to_mlflow.py                      # every finished run in runs/
    python scripts/log_to_mlflow.py --release checkpoints # also record the release as its own run
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse at http://127.0.0.1:5000

    # or against the tracking server of docker compose (docker compose --profile tracking up -d):
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/log_to_mlflow.py --release checkpoints

Training happens on several laptops and Kaggle/Colab, so runs are tracked on disk (runs/<name>/)
and imported here in one place. Each run gets its config as parameters, the per-epoch curves from
log.csv as stepped metrics, the test metrics of every dataset, its figures and tables as artifacts,
and tags for the experiment group (main / arch / loss / optim). Re-running skips runs that are
already imported (matched on the source_run tag), so it is safe to call after every new batch.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import yaml  # noqa: E402

from experiments import GROUPS  # noqa: E402
from src.tracking import git_commit, mlflow_client, params_of, test_metrics  # noqa: E402


def groups_of(name):
    """Experiment groups a run belongs to, and its label there."""
    found = {}
    for group, runs in GROUPS.items():
        for prefix, (label, *_) in runs.items():
            if name.rsplit("_s", 1)[0] == prefix:
                found[group] = label
    return found


def already_imported(mlflow):
    runs = mlflow.search_runs(output_format="list", max_results=5000)
    return {r.data.tags.get("source_run") for r in runs}


def import_run(mlflow, run_dir):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    groups = groups_of(run_dir.name)
    tags = {"source_run": run_dir.name, "imported_from": "runs/", "model": cfg["model"], "seed": str(cfg["seed"]),
            "groups": ",".join(groups) or "other"}
    tags |= {f"label_{g}": label for g, label in groups.items()}
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.set_tags(tags)
        mlflow.log_params(params_of(cfg))
        with open(run_dir / "log.csv") as f:
            for row in csv.DictReader(f):
                step = int(row.pop("epoch"))
                mlflow.log_metrics({k: float(v) for k, v in row.items() if v != ""}, step=step)
        for path in sorted(run_dir.glob("metrics_*.json")):
            mlflow.log_metrics(test_metrics(json.loads(path.read_text())))
        for pattern in ("config.yaml", "log.csv", "curves.png", "metrics_*.json", "per_image_*.csv"):
            for path in run_dir.glob(pattern):
                mlflow.log_artifact(str(path))


def log_release(mlflow, folder):
    meta = json.loads((folder / "registry.json").read_text())
    with mlflow.start_run(run_name=f"release-{meta['version']}"):
        mlflow.set_tags({"kind": "release", "version": meta["version"], "git_commit": meta.get("git_commit", git_commit()),
                         "default_model": meta["default"], "ensemble": ",".join(meta["ensemble"]["members"]),
                         "gate_passed": str(meta.get("gate", {}).get("passed", False))})
        for card in meta["models"]:
            mlflow.log_params({f"{card['id']}.source_run": card["source_run"], f"{card['id']}.sha256": card["sha256"][:16]})
            for dataset, values in card["metrics"].items():
                mlflow.log_metrics({f"{card['id']}.{dataset}_{k}": v for k, v in values.items() if k != "images"})
        for dataset, kinds in meta["ensemble"].get("metrics", {}).items():
            for kind, values in kinds.items():
                mlflow.log_metrics({f"ensemble_{kind}.{dataset}_{k}": v for k, v in values.items()})
        mlflow.log_artifact(str(folder / "registry.json"))
    print(f"Logged release {meta['version']} ({len(meta['models'])} models)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--release", help="release folder to record (e.g. checkpoints)")
    args = ap.parse_args()

    mlflow = mlflow_client()
    done = already_imported(mlflow)
    finished = sorted(p.parent for p in Path(args.runs_dir).glob("*/metrics_kvasir_test.json"))
    new = [r for r in finished if r.name not in done]
    for run_dir in new:
        import_run(mlflow, run_dir)
        print(f"imported {run_dir.name}")
    print(f"{len(new)} new runs imported, {len(finished) - len(new)} already present "
          f"-> {mlflow.get_tracking_uri()}")
    if args.release:
        log_release(mlflow, Path(args.release))


if __name__ == "__main__":
    main()
