"""Serving metrics shared by the Gradio UI and the REST API.

Prometheus scrapes GET /metrics (see deploy/prometheus.yml). Besides request counts and latency,
two histograms describe *what the models predict* (polyp area, highest probability) and a counter
records inputs flagged as unlike the training images. A shift in those distributions over time is
the first sign that the incoming images differ from Kvasir-SEG, even without labels.

If PREDICTION_LOG is set, one JSON line per prediction is appended to that file (summary numbers
only, never the image) for offline analysis.
"""
import json
import os
import threading
import time

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("polyp_requests_total", "Segmentation requests", ["model", "source", "status"])
LATENCY = Histogram("polyp_inference_seconds", "Model compute time per request", ["model"],
                    buckets=(0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2.5, 5, 10))
AREA = Histogram("polyp_predicted_area_ratio", "Fraction of the image predicted as polyp", ["model"],
                 buckets=(0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1))
MAX_PROB = Histogram("polyp_max_probability", "Highest polyp probability in the image", ["model"],
                     buckets=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1))
INPUT_WARNINGS = Counter("polyp_input_warnings_total", "Inputs flagged as unlike the training images", ["source"])
RELEASE = Gauge("polyp_release_info", "Model release served by this process", ["version", "commit"])

_log_lock = threading.Lock()


def set_release(meta):
    RELEASE.labels(meta.get("version", "?"), meta.get("git_commit", "?")).set(1)


def record(pred, source):
    """Record a finished Prediction (src.inference.Prediction)."""
    REQUESTS.labels(pred.model_id, source, "ok").inc()
    LATENCY.labels(pred.model_id).observe(pred.latency_ms / 1000)
    AREA.labels(pred.model_id).observe(float(pred.mask.mean()))
    MAX_PROB.labels(pred.model_id).observe(float(pred.prob.max()))
    if pred.warnings:
        INPUT_WARNINGS.labels(source).inc()
    path = os.environ.get("PREDICTION_LOG")
    if path:
        line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source, **pred.summary()})
        with _log_lock, open(path, "a") as f:
            f.write(line + "\n")


def record_error(model_id, source):
    REQUESTS.labels(model_id or "unknown", source, "error").inc()
