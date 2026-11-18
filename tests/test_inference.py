"""The serving path: release format, checksums, prediction at any image size, ensemble, input check."""
import json
import shutil

import numpy as np
import pytest

from src.inference import ENSEMBLE, Registry, find_regions, overlap_scores, predict


def test_registry_lists_models_and_ensemble(registry):
    assert registry.ids == ["tiny_a", "tiny_b"]
    assert registry.choices == ["tiny_a", "tiny_b", ENSEMBLE]
    assert registry.default == "tiny_a"


@pytest.mark.parametrize("shape", [(120, 200, 3), (333, 257, 3), (96, 96)])
def test_prediction_matches_input_size(registry, shape):
    rgb = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    pred = predict(registry, rgb, "tiny_a", threshold=0.5)
    assert pred.prob.shape == pred.mask.shape == shape[:2]
    assert pred.prob.min() >= 0 and pred.prob.max() <= 1
    assert set(np.unique(pred.mask)) <= {0, 1}
    assert pred.passes == 1 and pred.spread is None


def test_tta_and_ensemble_report_disagreement(registry):
    rgb = np.random.default_rng(1).integers(0, 255, (80, 100, 3), dtype=np.uint8)
    tta = predict(registry, rgb, "tiny_a", tta=True)
    ens = predict(registry, rgb, ENSEMBLE, tta=True)
    assert tta.passes == 4 and ens.passes == 8
    assert tta.spread.shape == rgb.shape[:2] and ens.spread.min() >= 0


def test_threshold_is_monotonic(registry):
    rgb = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    low, high = predict(registry, rgb, "tiny_b", threshold=0.1), predict(registry, rgb, "tiny_b", threshold=0.9)
    assert low.mask.sum() >= high.mask.sum()


def test_unknown_model_is_rejected(registry):
    with pytest.raises(KeyError):
        predict(registry, np.zeros((64, 64, 3), np.uint8), "nope")


def test_corrupted_file_is_detected(release, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(release, broken)
    with open(broken / "tiny_a.pt", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(RuntimeError, match="Checksum"):
        Registry(broken).model("tiny_a")


def test_missing_release_explains_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="package_models"):
        Registry(tmp_path)


def test_small_regions_are_removed():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:40, 10:40] = 1  # 9 %
    mask[80:82, 80:82] = 1  # 0.04 %
    prob = mask.astype(np.float32)
    regions = find_regions(mask, prob, min_area_pct=0.1)
    assert len(regions) == 1 and regions[0].area_px == 900
    assert mask.sum() == 900


def test_overlap_scores():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    s = overlap_scores(a, a)
    assert s["dice"] == pytest.approx(1) and s["iou"] == pytest.approx(1)
    assert overlap_scores(a, ~a)["dice"] < 1e-6


def test_input_check_flags_unusual_images(release, tmp_path):
    meta = json.loads((release / "registry.json").read_text())
    meta["input_stats"] = {"mean": [140, 80, 60, 130, 58], "std": [20, 12, 12, 18, 8], "z_limit": 4}
    folder = tmp_path / "with_stats"
    shutil.copytree(release, folder)
    (folder / "registry.json").write_text(json.dumps(meta))
    reg = Registry(folder)
    reddish = np.zeros((200, 200, 3), np.uint8)  # mucosa-like colours with some contrast
    reddish[:100] = (90, 50, 38)
    reddish[100:] = (195, 110, 82)
    blue = np.zeros((200, 200, 3), np.uint8)
    blue[...] = (30, 60, 220)
    assert not reg.check_input(reddish)
    assert any("Colour" in w for w in reg.check_input(blue))
    assert any("small" in w for w in reg.check_input(np.zeros((60, 60, 3), np.uint8)))
