"""Experiment tracking for src/train.py: Weights & Biases and/or MLflow, both optional.

    python -m src.train --config configs/unet_resnet34.yaml --mlflow     # local: mlruns/mlflow.db
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train ... --mlflow   # docker compose server

Runs trained without a tracker can be imported afterwards with scripts/log_to_mlflow.py, which
writes the same parameters, per-epoch metrics, test metrics and artifacts.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "polyp-segmentation"
LOCAL_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"
ARTIFACT_FIELDS = ("config.yaml", "log.csv", "curves.png")


def mlflow_client():
    """mlflow module pointed at MLFLOW_TRACKING_URI, or a local SQLite store under mlruns/."""
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        (ROOT / "mlruns").mkdir(exist_ok=True)
        uri = LOCAL_URI
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        location = None if uri.startswith("http") else (ROOT / "mlruns" / "artifacts").as_uri()
        mlflow.create_experiment(EXPERIMENT, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def params_of(cfg):
    return {k: ("none" if v is None else v) for k, v in cfg.items() if k not in ("wandb", "mlflow", "out_dir")}


def test_metrics(summary):
    """Flatten an evaluate_run summary into {"<dataset>_<metric>": value}."""
    out = {f"{summary['dataset']}_{k}": v for k, v in summary["mean"].items()}
    for device, ms in summary.get("latency_ms", {}).items():
        out[f"latency_{device}_ms"] = ms
    out["params_millions"] = summary["params_millions"]
    return out


class Tracker:
    def __init__(self, cfg, tags=None):
        self.wandb = self.mlflow = None
        if cfg.get("wandb"):
            import wandb
            self.wandb = wandb.init(project=EXPERIMENT, name=cfg["name"], config=cfg)
        if cfg.get("mlflow"):
            self.mlflow = mlflow_client()
            self.mlflow.start_run(run_name=cfg["name"])
            self.mlflow.set_tags({"git_commit": git_commit(), "source_run": cfg["name"], **(tags or {})})
            self.mlflow.log_params(params_of(cfg))

    def log_epoch(self, row):
        if self.wandb:
            self.wandb.log(row)
        if self.mlflow:
            self.mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])

    def log_test(self, summary, run_dir=None):
        metrics = test_metrics(summary)
        if self.wandb:
            self.wandb.summary.update({f"test_{k}": v for k, v in summary["mean"].items()})
        if self.mlflow:
            self.mlflow.log_metrics(metrics)
            if run_dir:
                for name in ARTIFACT_FIELDS:
                    if (Path(run_dir) / name).exists():
                        self.mlflow.log_artifact(str(Path(run_dir) / name))

    def finish(self):
        if self.wandb:
            self.wandb.finish()
        if self.mlflow:
            self.mlflow.end_run()
