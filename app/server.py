"""HTTP server: Gradio UI at /, REST API at /api/v1, health at /health, Prometheus at /metrics.

    python -m app.server                         # 127.0.0.1:7860
    HOST=0.0.0.0 PORT=8000 python -m app.server

Environment:
    POLYP_MODEL_DIR   release folder (default checkpoints/)
    WARMUP            models loaded at start-up: default | all | none   (default: default)
    MAX_UPLOAD_MB     largest accepted image (default 15)
    PREDICTION_LOG    optional JSONL file with one summary line per prediction
"""
import base64
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: E402

from app import monitoring  # noqa: E402
from app.app import EXAMPLES_DIR, THEME, build_demo, css  # noqa: E402
from src.inference import ENSEMBLE, Registry, predict  # noqa: E402

MODEL_DIR = os.environ.get("POLYP_MODEL_DIR", str(ROOT / "checkpoints"))
MAX_UPLOAD = int(float(os.environ.get("MAX_UPLOAD_MB", 15)) * 2**20)


def create_app(model_dir=MODEL_DIR, warmup=None):
    registry = Registry(model_dir)
    monitoring.set_release(registry.meta)
    warmup = warmup or os.environ.get("WARMUP", "default")
    started = time.time()

    @asynccontextmanager
    async def lifespan(_):
        if warmup != "none":
            ids = registry.ids if warmup == "all" else [registry.default]
            await run_in_threadpool(registry.warmup, ids)
        yield

    api = FastAPI(title="Polyp segmentation API", version=str(registry.meta.get("version")), lifespan=lifespan,
                  description="Segments polyps in colonoscopy frames. Research use only, not a medical device.")

    @api.get("/health", tags=["ops"])
    def health():
        return {"status": "ok", "release": registry.meta.get("version"), "commit": registry.meta.get("git_commit"),
                "models": registry.choices, "loaded": sorted(registry._models), "uptime_s": round(time.time() - started)}

    @api.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @api.get("/api/v1/models", tags=["inference"])
    def models():
        keep = ("id", "label", "architecture", "encoder", "pretrained", "params_m", "cpu_ms", "img_size",
                "metrics", "source_run", "sha256")
        return {"release": registry.meta.get("version"), "default": registry.default,
                "models": [{k: c.get(k) for k in keep} for c in registry.meta["models"]],
                "ensemble": registry.meta.get("ensemble")}

    async def run(file, model, tta, threshold, min_area_pct):
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"Image larger than {MAX_UPLOAD // 2**20} MB")
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(415, "Could not decode the upload as an image (JPEG, PNG, BMP, TIFF, WebP).")
        if model not in registry.choices:
            raise HTTPException(404, f"Unknown model '{model}'. Available: {registry.choices}")
        try:
            pred = await run_in_threadpool(predict, registry, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), model, tta,
                                           threshold, min_area_pct)
        except Exception as e:
            monitoring.record_error(model, "api")
            raise HTTPException(500, f"Inference failed: {e}") from e
        monitoring.record(pred, "api")
        return pred

    model_q = Query(None, description=f"model id or '{ENSEMBLE}'; default: the release default")
    tta_q = Query(False, description="average over 4 flipped copies")
    thr_q = Query(0.5, ge=0.01, le=0.99, description="probability threshold")
    area_q = Query(0.0, ge=0, le=50, description="drop regions smaller than this % of the image")

    @api.post("/api/v1/predict", tags=["inference"])
    async def predict_json(file: UploadFile = File(...), model: str | None = model_q, tta: bool = tta_q,
                           threshold: float = thr_q, min_area_pct: float = area_q,
                           include_mask: bool = Query(False, description="add the mask as a base64 PNG")):
        pred = await run(file, model or registry.default, tta, threshold, min_area_pct)
        body = {**pred.summary(), "image_size": {"width": pred.mask.shape[1], "height": pred.mask.shape[0]}}
        if include_mask:
            body["mask_png_base64"] = base64.b64encode(cv2.imencode(".png", pred.mask * 255)[1]).decode()
        return body

    @api.post("/api/v1/predict/mask", tags=["inference"], response_class=Response,
              responses={200: {"content": {"image/png": {}}}})
    async def predict_mask(file: UploadFile = File(...), model: str | None = model_q, tta: bool = tta_q,
                           threshold: float = thr_q, min_area_pct: float = area_q):
        pred = await run(file, model or registry.default, tta, threshold, min_area_pct)
        return Response(cv2.imencode(".png", pred.mask * 255)[1].tobytes(), media_type="image/png",
                        headers={"X-Polyp-Area-Pct": f"{pred.area_pct:.3f}", "X-Model": pred.model_id,
                                 "X-Latency-Ms": f"{pred.latency_ms:.1f}"})

    demo = build_demo(registry)
    demo.queue(default_concurrency_limit=2)
    return gr.mount_gradio_app(api, demo, path="/", theme=THEME, css=css(), allowed_paths=[str(EXAMPLES_DIR)],
                               footer_links=["api", "settings"])


def main():
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
