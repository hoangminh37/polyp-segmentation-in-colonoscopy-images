"""Send images to the running API: load test, and a reproducible input-drift scenario for the dashboard.

    python scripts/load_test.py data/raw/Kvasir-SEG/images --n 200 --workers 4
    python scripts/load_test.py data/raw/CVC-ClinicDB/images --model ensemble
    python scripts/load_test.py --drift --n 150      # synthetic non-endoscopic images

Prints throughput, latency percentiles (end-to-end and model-only) and how many inputs the
server flagged. With `docker compose --profile monitoring up`, the effect is visible in Grafana:
`--drift` raises "Flagged inputs" and moves the predicted-area and peak-probability panels.
"""
import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def drift_image(rng):
    """Images unlike colonoscopy frames: flat colours, gradients, noise, grayscale."""
    h, w = rng.integers(240, 600, 2)
    kind = rng.integers(4)
    if kind == 0:
        img = np.full((h, w, 3), rng.integers(0, 255, 3), np.uint8)
    elif kind == 1:
        ramp = np.linspace(0, 255, w, dtype=np.uint8)
        img = np.stack([np.tile(ramp, (h, 1))] * 3, -1)
        img[..., rng.integers(3)] = 255 - img[..., 0]
    elif kind == 2:
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        gray = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 8)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.imencode(".png", img)[1].tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", help="folders of images to send")
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--model", help="model id or 'ensemble' (default: the release default)")
    ap.add_argument("--n", type=int, default=100, help="number of requests")
    ap.add_argument("--workers", type=int, default=2, help="concurrent requests")
    ap.add_argument("--drift", action="store_true", help="send synthetic non-endoscopic images")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    paths = [p for f in args.folders for p in sorted(Path(f).iterdir()) if p.suffix.lower() in EXTS]
    if not paths and not args.drift:
        ap.error("give at least one image folder or --drift")
    random.Random(0).shuffle(paths)
    payloads = [drift_image(rng) if args.drift else paths[i % len(paths)].read_bytes() for i in range(args.n)]
    params = {"model": args.model} if args.model else {}

    def send(data):
        start = time.perf_counter()
        r = requests.post(f"{args.url}/api/v1/predict", params=params, files={"file": ("image", data)}, timeout=120)
        total = (time.perf_counter() - start) * 1000
        body = r.json() if r.ok else {}
        return r.status_code, total, body.get("latency_ms"), bool(body.get("input_warnings"))

    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(send, payloads))
    elapsed = time.perf_counter() - start

    ok = [r for r in results if r[0] == 200]
    total = sorted(r[1] for r in ok)
    model = sorted(r[2] for r in ok)

    def pct(values, q):
        return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")

    print(f"{len(ok)}/{len(results)} ok in {elapsed:.1f} s  ({len(results) / elapsed:.1f} req/s, {args.workers} workers)")
    if ok:
        print(f"end-to-end ms   p50 {pct(total, .5):7.0f}   p95 {pct(total, .95):7.0f}   mean {statistics.mean(total):7.0f}")
        print(f"model-only ms   p50 {pct(model, .5):7.0f}   p95 {pct(model, .95):7.0f}")
        print(f"flagged inputs  {sum(r[3] for r in ok)}/{len(ok)}")
    errors = {r[0] for r in results if r[0] != 200}
    if errors:
        print(f"HTTP errors: {sorted(errors)}")


if __name__ == "__main__":
    main()
