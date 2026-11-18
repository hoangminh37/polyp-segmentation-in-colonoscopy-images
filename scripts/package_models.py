"""Build a model release for the demo and the API: checkpoints/registry.json + one file per model.

    python scripts/package_models.py                     # all models of the main comparison
    python scripts/package_models.py --models unet_r34 segformer_b2 --default segformer_b2
    python scripts/package_models.py --dummy --out /tmp/release   # random tiny models, no data needed (CI)

For every model prefix it
  1. picks the seed whose checkpoint has the best *validation* Dice (the test split is not used
     to choose, so the reported test Dice stays unbiased),
  2. stores the weights in float16 with a SHA-256 checksum,
  3. copies the run's recorded test metrics, parameter count and CPU latency,
  4. release gate: re-evaluates the stored file on the Kvasir test split and stops if its Dice
     differs from the recorded one by more than --tolerance (catches a wrong file, a broken
     conversion or preprocessing that drifted from training),
and then measures the ensemble (with and without flip TTA) on every available test set,
records colour statistics of the training images for the demo's input check, and copies
example images with their ground-truth masks to app/examples/.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import with_defaults  # noqa: E402
from src.inference import REGISTRY_FILE, Registry, image_features, sha256  # noqa: E402
from src.models import build_model, count_params, get_device  # noqa: E402

MODELS = {  # prefix -> (label, short name)
    "custom_unet": ("Custom U-Net (from scratch)", "Custom U-Net"),
    "unet_r34_scratch": ("U-Net · ResNet-34 (no pretraining)", "U-Net scratch"),
    "unet_r34": ("U-Net · ResNet-34", "U-Net"),
    "deeplabv3plus_r34": ("DeepLabV3+ · ResNet-34", "DeepLabV3+"),
    "segformer_b2": ("SegFormer · MiT-B2", "SegFormer"),
}
DEFAULT_MODELS = ["custom_unet", "unet_r34", "deeplabv3plus_r34", "segformer_b2"]
DEFAULT_ENSEMBLE = ["unet_r34", "deeplabv3plus_r34", "segformer_b2"]


def git_version():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pick_run(runs_dir, prefix, seed=None):
    """Run directory of `prefix` with the highest validation Dice (or the given seed)."""
    candidates = [runs_dir / f"{prefix}_s{seed}"] if seed is not None else sorted(runs_dir.glob(f"{prefix}_s*"))
    candidates = [c for c in candidates if (c / "best.pt").exists() and (c / "metrics_kvasir_test.json").exists()
                  and c.name.removeprefix(prefix + "_s").isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No finished run for '{prefix}' in {runs_dir}")
    scored = [(torch.load(c / "best.pt", map_location="cpu", weights_only=True)["val_dice"], c) for c in candidates]
    return max(scored, key=lambda s: s[0])[1]


def save_half(ckpt, path):
    """Store floating-point tensors as float16 (BatchNorm counters stay int64)."""
    state = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
    torch.save({"model": state, "cfg": ckpt["cfg"], "epoch": ckpt.get("epoch"), "val_dice": ckpt.get("val_dice")}, path)


def model_card(prefix, run_dir, file, out):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = with_defaults(ckpt["cfg"])
    save_half(ckpt, out / file)
    metrics = {}
    for path in sorted(run_dir.glob("metrics_*.json")):
        m = json.loads(path.read_text())
        metrics[m["dataset"]] = {k: round(v, 4) for k, v in m["mean"].items()} | {"images": m["num_images"]}
    recorded = json.loads((run_dir / "metrics_kvasir_test.json").read_text())
    label, short = MODELS.get(prefix, (prefix, prefix))
    return {
        "id": prefix, "label": label, "short": short, "file": file, "sha256": sha256(out / file),
        "size_mb": round((out / file).stat().st_size / 2**20, 1),
        "source_run": run_dir.name, "epoch": ckpt.get("epoch"), "val_dice": round(ckpt["val_dice"], 4),
        "architecture": cfg["model"], "encoder": None if cfg["model"] == "custom_unet" else cfg["encoder"],
        "pretrained": cfg["model"] != "custom_unet" and cfg["encoder_weights"] not in (None, "", "none"),
        "img_size": cfg["img_size"], "loss": cfg["loss"],
        "params_m": round(recorded["params_millions"], 1),
        "cpu_ms": round(recorded["latency_ms"].get("cpu", float("nan")), 1),
        "metrics": metrics,
    }


def input_stats(cfg):
    """Mean/std of image_features over the training split (used to warn on unusual inputs)."""
    from src.data import list_pairs, read_split
    train = set(read_split(cfg["split_dir"], cfg["dataset_name"], "train"))
    feats = []
    for stem, img_path, _ in list_pairs(cfg["data_dir"]):
        if stem in train:
            rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            feats.append(image_features(rgb))
    feats = np.stack(feats)
    return {"features": ["mean_r", "mean_g", "mean_b", "saturation", "contrast"],
            "mean": feats.mean(0).round(3).tolist(), "std": feats.std(0).round(3).tolist(),
            "z_limit": 4.0, "images": len(feats)}


def evaluation_sets(cfg):
    from experiments import EXTERNAL_DATASETS
    from src.data import PolypDataset, load_arrays, read_split
    names, images, masks = load_arrays(cfg["data_dir"], cfg["img_size"])
    sets = {"kvasir_test": PolypDataset(names, images, masks, read_split(cfg["split_dir"], cfg["dataset_name"], "test"))}
    for name, path in EXTERNAL_DATASETS.items():
        if (ROOT / path).exists():
            sets[name] = PolypDataset(*load_arrays(path, cfg["img_size"]))
    return {k: (torch.stack([d[i][0] for i in range(len(d))]), torch.stack([d[i][1] for i in range(len(d))]))
            for k, d in sets.items()}


def release_gate(registry, cards, ensemble, tolerance):
    """Re-evaluate the stored files; return ensemble metrics. Exits on a regression."""
    from src.ensemble import predict_probs, score
    sets = evaluation_sets(with_defaults({}))
    x, y = sets["kvasir_test"]
    failed = False
    for card in cards:
        dice = score(predict_probs(registry.model(card["id"]), x, registry.device, tta=False), y)["dice"]
        recorded = card["metrics"]["kvasir_test"]["dice"]
        ok = abs(dice - recorded) <= tolerance
        failed |= not ok
        print(f"  gate {card['id']:<20} recorded Dice {recorded:.4f}  packaged {dice:.4f}  {'ok' if ok else 'FAIL'}")
    if failed:
        sys.exit(f"Release gate failed: a packaged model does not reproduce its recorded Dice (±{tolerance}).")
    results = {}
    if len(ensemble) > 1:
        for name, (x, y) in sets.items():
            for tta in (False, True):
                probs = sum(predict_probs(registry.model(m), x, registry.device, tta).float() for m in ensemble)
                m = score(probs / len(ensemble), y)
                results.setdefault(name, {})["tta" if tta else "plain"] = {k: round(v, 4) for k, v in m.items()}
            print(f"  ensemble on {name}: Dice {results[name]['plain']['dice']:.4f}, "
                  f"with TTA {results[name]['tta']['dice']:.4f}")
    return results


def pick_examples(card, cfg, out_dir):
    """Copy a few test images with their masks, chosen to show typical, hard and external cases."""
    import pandas as pd
    from src.data import list_pairs
    run = ROOT / "runs" / card["source_run"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "masks").mkdir()

    def components(mask_path):
        """Polyp area fraction and number of polyps covering at least 2% of the image."""
        m = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        big = [s for s in stats[1:, 4] if s / m.size >= 0.02]
        return m.mean(), len(big)

    chosen = []  # (stem, label, note, source, pairs dict)
    kvasir = {s: (i, m) for s, i, m in list_pairs(cfg["data_dir"])}
    df = pd.read_csv(run / "per_image_kvasir_test.csv", dtype={"image": str})
    df[["area", "parts"]] = [components(kvasir[s][1]) for s in df["image"]]
    good = df[df["dice"] >= 0.85]
    picks = [
        (good.sort_values("area").iloc[0], "Small polyp"),
        (good.iloc[(good["area"] - good["area"].median()).abs().argsort().iloc[0]], "Typical polyp"),
        (good.sort_values("area").iloc[-1], "Large polyp"),
        (df[df["parts"] >= 2].sort_values("dice").iloc[-1] if (df["parts"] >= 2).any() else None, "Two polyps"),
        (df.sort_values("dice").iloc[0], "Hard case"),
    ]
    used = set()
    for row, label in picks:
        if row is not None and row["image"] not in used:
            used.add(row["image"])
            note = f"{card['short']} Dice {row['dice']:.2f} at release; polyp covers {row['area'] * 100:.1f}% of the frame"
            chosen.append((row["image"], label, note, "Kvasir-SEG test split", kvasir))

    clinic_csv = run / "per_image_clinicdb.csv"
    if clinic_csv.exists():
        from experiments import EXTERNAL_DATASETS
        clinic = {s: (i, m) for s, i, m in list_pairs(ROOT / EXTERNAL_DATASETS["clinicdb"])}
        cd = pd.read_csv(clinic_csv, dtype={"image": str})
        cd = cd[cd["image"].isin(clinic)]
        median = cd.iloc[(cd["dice"] - cd["dice"].median()).abs().argsort().iloc[0]]
        note = f"Different hospital and camera; {card['short']} Dice {median['dice']:.2f} (median on ClinicDB)"
        chosen.append((median["image"], "Other hospital", note, "CVC-ClinicDB, external", clinic))

    entries = []
    for stem, label, note, source, pairs in chosen:
        img_path, mask_path = pairs[stem]
        prefix = "clinicdb_" if "Clinic" in source else ""
        img_name = f"{prefix}{stem}{img_path.suffix.lower()}"
        shutil.copy(img_path, out_dir / "images" / img_name)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / "masks" / f"{prefix}{stem}.png"), mask)
        entries.append({"image": f"images/{img_name}", "mask": f"masks/{prefix}{stem}.png",
                        "label": label, "note": note, "source": source})
    (out_dir / "examples.json").write_text(json.dumps(entries, indent=2))
    return entries


def build_dummy_release(out, img_size=64):
    """Two tiny random-weight models: lets tests and CI exercise the full serving path without data."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, model_id in enumerate(["tiny_a", "tiny_b"]):
        torch.manual_seed(i)
        cfg = with_defaults({"model": "custom_unet", "base_channels": 4, "img_size": img_size, "name": model_id})
        model = build_model(cfg)
        save_half({"model": model.state_dict(), "cfg": cfg}, out / f"{model_id}.pt")
        cards.append({"id": model_id, "label": f"Tiny U-Net {i + 1}", "short": f"Tiny {i + 1}",
                      "file": f"{model_id}.pt", "sha256": sha256(out / f"{model_id}.pt"),
                      "source_run": "dummy", "architecture": "custom_unet", "encoder": None, "pretrained": False,
                      "img_size": img_size, "params_m": round(count_params(model) / 1e6, 3), "cpu_ms": None,
                      "val_dice": None, "metrics": {}})
    meta = {"version": "dummy", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_version(),
            "models": cards, "default": "tiny_a", "ensemble": {"members": ["tiny_a", "tiny_b"], "metrics": {}},
            "input_stats": None, "dummy": True}
    (out / REGISTRY_FILE).write_text(json.dumps(meta, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="run prefixes to package")
    ap.add_argument("--ensemble", nargs="+", default=DEFAULT_ENSEMBLE, help="prefixes averaged by the ensemble")
    ap.add_argument("--default", default="segformer_b2", help="model selected when the demo opens")
    ap.add_argument("--seed", type=int, help="use this seed for every model instead of the best validation Dice")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--examples-dir", default=str(ROOT / "app" / "examples"))
    ap.add_argument("--tolerance", type=float, default=0.005, help="max allowed Dice change in the release gate")
    ap.add_argument("--skip-gate", action="store_true", help="do not re-evaluate (faster, not recommended)")
    ap.add_argument("--dummy", action="store_true", help="write a tiny random release (no data or runs needed)")
    ap.add_argument("--examples-only", action="store_true", help="only re-pick the example images of the current release")
    args = ap.parse_args()

    out = Path(args.out)
    if args.dummy:
        print(f"Dummy release written to {build_dummy_release(out)}")
        return
    if args.examples_only:
        meta = json.loads((out / REGISTRY_FILE).read_text())
        card = next(c for c in meta["models"] if c["id"] == meta["default"])
        examples = pick_examples(card, with_defaults({}), Path(args.examples_dir))
        print("\n".join(f"{e['label']:<16} {e['image']}  ({e['note']})" for e in examples))
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cards = []
    for prefix in args.models:
        run_dir = pick_run(Path(args.runs_dir), prefix, args.seed)
        cards.append(model_card(prefix, run_dir, f"{prefix}.pt", out))
        c = cards[-1]
        print(f"{prefix:<20} <- {run_dir.name:<24} val Dice {c['val_dice']:.4f}  "
              f"test Dice {c['metrics']['kvasir_test']['dice']:.4f}  {c['size_mb']} MB")
    ensemble = [m for m in args.ensemble if m in args.models]
    cfg = with_defaults({})
    meta = {
        "version": time.strftime("%Y.%m.%d"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_version(), "torch": torch.__version__,
        "models": cards, "default": args.default if args.default in args.models else args.models[-1],
        "ensemble": {"members": ensemble, "metrics": {}},
        "evaluation": "Dice/IoU/precision/recall per image at 352×352, threshold 0.5, averaged over images",
        "input_stats": input_stats(cfg),
    }
    (out / REGISTRY_FILE).write_text(json.dumps(meta, indent=2))

    if not args.skip_gate:
        print("Release gate (re-evaluating the packaged files):")
        registry = Registry(out, device=get_device())
        meta["ensemble"]["metrics"] = release_gate(registry, cards, ensemble, args.tolerance)
        meta["gate"] = {"passed": True, "tolerance": args.tolerance}
        (out / REGISTRY_FILE).write_text(json.dumps(meta, indent=2))

    default_card = next(c for c in cards if c["id"] == meta["default"])
    examples = pick_examples(default_card, cfg, Path(args.examples_dir))
    total = sum(c["size_mb"] for c in cards)
    print(f"\nRelease {meta['version']} ({meta['git_commit']}): {len(cards)} models, {total:.0f} MB -> {out}/")
    print(f"{len(examples)} example images with masks -> {args.examples_dir}/")


if __name__ == "__main__":
    main()
