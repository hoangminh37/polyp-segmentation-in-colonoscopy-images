"""REST API and UI mounted on one FastAPI app, served from a dummy release."""
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(release):
    from app.server import create_app
    with TestClient(create_app(release, warmup="all")) as c:
        yield c


def _png(h=90, w=120):
    rgb = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".png", rgb)[1].tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["release"] == "dummy"
    assert set(r.json()["loaded"]) == {"tiny_a", "tiny_b"}


def test_models(client):
    body = client.get("/api/v1/models").json()
    assert [m["id"] for m in body["models"]] == ["tiny_a", "tiny_b"]


def test_predict_json_with_mask(client):
    r = client.post("/api/v1/predict", params={"model": "ensemble", "tta": True, "include_mask": True},
                    files={"file": ("frame.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "ensemble" and body["forward_passes"] == 8
    assert body["image_size"] == {"width": 120, "height": 90}
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(body["mask_png_base64"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (90, 120)


def test_predict_mask_png(client):
    r = client.post("/api/v1/predict/mask", files={"file": ("frame.png", _png(64, 80), "image/png")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "x-polyp-area-pct" in r.headers
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE).shape == (64, 80)


@pytest.mark.parametrize("params,payload,status", [
    ({}, b"not an image", 415),
    ({"model": "unknown"}, None, 404),
    ({"threshold": 1.5}, None, 422),
])
def test_bad_requests(client, params, payload, status):
    r = client.post("/api/v1/predict", params=params, files={"file": ("x.png", payload or _png(), "image/png")})
    assert r.status_code == status


def test_metrics_count_requests(client):
    client.post("/api/v1/predict", files={"file": ("frame.png", _png(), "image/png")})
    text = client.get("/metrics").text
    assert 'polyp_requests_total{model="tiny_a",source="api",status="ok"}' in text
    assert "polyp_inference_seconds_bucket" in text


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
