"""Inference used by the demo and the REST API.

A *release* is a folder written by scripts/package_models.py:

    checkpoints/
      registry.json        models, their test metrics, checksums, input statistics, ensemble
      custom_unet.pt       weights stored in float16 (half the size; Dice change < 0.001)
      unet_r34.pt
      ...

`Registry` reads that folder and loads each model on first use. `predict` runs one model or the
ensemble on an RGB image of any size and returns the probability map and mask at the original
resolution, the connected regions, and (when several predictions are averaged) how much they
disagree.
"""
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import preprocess_image
from .models import build_model

REGISTRY_FILE = "registry.json"
ENSEMBLE = "ensemble"
FLIPS = [[], [3], [2], [2, 3]]  # same test-time flips as src/ensemble.py: none, h, v, both


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def image_features(rgb):
    """Cheap per-image statistics used to flag inputs unlike the training images:
    mean R, G, B, mean saturation and contrast (std of gray level), all on a 0-255 scale."""
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return np.array([*small.reshape(-1, 3).mean(0), hsv[..., 1].mean(), gray.std()], np.float64)


FEATURE_NAMES = ["mean red", "mean green", "mean blue", "saturation", "contrast"]


@dataclass
class Region:
    area_px: int
    area_pct: float
    bbox: tuple  # x, y, width, height in original pixels
    mean_prob: float


@dataclass
class Prediction:
    prob: np.ndarray  # H×W float32, probability of polyp at the original resolution
    mask: np.ndarray  # H×W uint8 in {0, 1}
    spread: np.ndarray | None  # H×W std of the averaged predictions, None for a single pass
    regions: list
    latency_ms: float
    model_id: str
    passes: int  # forward passes averaged (models × flips)
    threshold: float
    warnings: list = field(default_factory=list)

    @property
    def area_pct(self):
        return float(self.mask.mean() * 100)

    def summary(self):
        return {
            "model": self.model_id,
            "threshold": self.threshold,
            "forward_passes": self.passes,
            "latency_ms": round(self.latency_ms, 1),
            "polyp_area_pct": round(self.area_pct, 3),
            "max_probability": round(float(self.prob.max()), 4),
            "regions": [{"area_px": r.area_px, "area_pct": round(r.area_pct, 3), "bbox_xywh": list(r.bbox),
                         "mean_probability": round(r.mean_prob, 4)} for r in self.regions],
            "input_warnings": self.warnings,
        }


class Registry:
    """The models of one release. Thread-safe lazy loading; models stay in memory once loaded."""

    def __init__(self, model_dir, device="cpu", verify=True):
        self.dir = Path(model_dir)
        path = self.dir / REGISTRY_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Build a release first: "
                                    f"python scripts/package_models.py")
        self.meta = json.loads(path.read_text())
        self.cards = {m["id"]: m for m in self.meta["models"]}
        self.device = torch.device(device)
        self.verify = verify
        self._models, self._lock = {}, threading.Lock()

    # --- metadata -------------------------------------------------------------------------
    @property
    def ids(self):
        return list(self.cards)

    @property
    def ensemble_members(self):
        return [m for m in self.meta.get("ensemble", {}).get("members", []) if m in self.cards]

    @property
    def choices(self):
        """Model ids offered to users, ensemble last."""
        return self.ids + ([ENSEMBLE] if len(self.ensemble_members) > 1 else [])

    @property
    def default(self):
        return self.meta.get("default") or self.ids[-1]

    def label(self, model_id):
        if model_id == ENSEMBLE:
            return "Ensemble (" + " + ".join(self.cards[m]["short"] for m in self.ensemble_members) + ")"
        return self.cards[model_id]["label"]

    # --- loading --------------------------------------------------------------------------
    def model(self, model_id):
        if model_id not in self.cards:
            raise KeyError(f"Unknown model '{model_id}'. Available: {self.choices}")
        with self._lock:
            if model_id not in self._models:
                self._models[model_id] = self._load(self.cards[model_id])
        return self._models[model_id]

    def _load(self, card):
        path = self.dir / card["file"]
        if self.verify and card.get("sha256") and sha256(path) != card["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {path}; the release folder is corrupted.")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(ckpt["cfg"], pretrained=False)
        state = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        return model.to(self.device).eval()

    def warmup(self, model_ids=None):
        """Load models and run one dummy pass each, so the first real request is not slow."""
        for model_id in model_ids or self.ids:
            size = self.cards[model_id]["img_size"]
            with torch.inference_mode():
                self.model(model_id)(torch.zeros(1, 3, size, size, device=self.device))

    # --- input check ----------------------------------------------------------------------
    def check_input(self, rgb):
        """Return human-readable reasons why an image may be outside what the models were trained on."""
        warnings = []
        h, w = rgb.shape[:2]
        if min(h, w) < 128:
            warnings.append(f"The image is small ({w}×{h}); training images were at least 332 pixels wide.")
        stats = self.meta.get("input_stats")
        if stats:
            z = (image_features(rgb) - np.array(stats["mean"])) / np.maximum(np.array(stats["std"]), 1e-6)
            limit = stats.get("z_limit", 4.0)
            off = [f"{name} ({'high' if v > 0 else 'low'})" for name, v in zip(FEATURE_NAMES, z) if abs(v) > limit]
            if off:
                warnings.append("Colour statistics differ from the colonoscopy training images: "
                                + ", ".join(off) + ". The prediction may be unreliable.")
        return warnings


@torch.inference_mode()
def _probabilities(model, x, tta):
    """[passes, S, S] probabilities, one per flip (or a single pass)."""
    out = []
    for dims in (FLIPS if tta else FLIPS[:1]):
        xi = x.flip(dims) if dims else x
        p = torch.sigmoid(model(xi).float())
        out.append((p.flip(dims) if dims else p)[0, 0])
    return torch.stack(out).cpu().numpy()


def find_regions(mask, prob, min_area_pct=0.0):
    """Connected components of the mask, largest first. Components smaller than min_area_pct
    (percent of the image) are removed from the mask in place."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = mask.size
    regions = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area / total * 100 < min_area_pct:
            mask[labels == i] = 0
            continue
        regions.append(Region(area, area / total * 100, (x, y, w, h), float(prob[labels == i].mean())))
    return sorted(regions, key=lambda r: -r.area_px)


def as_rgb(image):
    """Accept gray, RGB or RGBA uint8 arrays; return contiguous RGB."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(image[..., :3])


@dataclass
class RawPrediction:
    """Model output before thresholding, so the threshold can change without re-running models."""
    prob: np.ndarray
    spread: np.ndarray | None
    latency_ms: float
    model_id: str
    passes: int
    warnings: list


def predict_raw(registry, rgb, model_id=None, tta=False):
    rgb = as_rgb(rgb)
    model_id = model_id or registry.default
    if model_id != ENSEMBLE and model_id not in registry.cards:
        raise KeyError(f"Unknown model '{model_id}'. Available: {registry.choices}")
    members = registry.ensemble_members if model_id == ENSEMBLE else [model_id]
    models = [registry.model(m) for m in members]  # load before timing

    start = time.perf_counter()
    probs = []
    for member, model in zip(members, models):
        x = preprocess_image(rgb, registry.cards[member]["img_size"]).to(registry.device)
        probs.append(_probabilities(model, x, tta))
    probs = np.concatenate(probs)  # [passes, S, S]
    latency = (time.perf_counter() - start) * 1000

    h, w = rgb.shape[:2]
    prob = cv2.resize(probs.mean(0), (w, h), interpolation=cv2.INTER_LINEAR)
    spread = cv2.resize(probs.std(0), (w, h), interpolation=cv2.INTER_LINEAR) if len(probs) > 1 else None
    return RawPrediction(prob, spread, latency, model_id, len(probs), registry.check_input(rgb))


def finalize(raw, threshold=0.5, min_area_pct=0.0):
    mask = (raw.prob > threshold).astype(np.uint8)
    regions = find_regions(mask, raw.prob, min_area_pct)
    return Prediction(raw.prob, mask, raw.spread, regions, raw.latency_ms, raw.model_id, raw.passes,
                      threshold, raw.warnings)


def predict(registry, rgb, model_id=None, tta=False, threshold=0.5, min_area_pct=0.0):
    """Segment one RGB uint8 image (H×W×3, any size)."""
    return finalize(predict_raw(registry, rgb, model_id, tta), threshold, min_area_pct)


def overlap_scores(mask, truth):
    """Dice, IoU, precision and recall of a predicted mask against a ground-truth mask (same size)."""
    p, t = mask.astype(bool), truth.astype(bool)
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    eps = 1e-7
    return {"dice": (2 * tp + eps) / (2 * tp + fp + fn + eps), "iou": (tp + eps) / (tp + fp + fn + eps),
            "precision": (tp + eps) / (tp + fp + eps), "recall": (tp + eps) / (tp + fn + eps)}
