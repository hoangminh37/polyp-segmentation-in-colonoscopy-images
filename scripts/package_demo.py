"""Prepare the demo: copy a trained checkpoint to checkpoints/model.pt and pick example images.

    python scripts/package_demo.py --run runs/unet_resnet34_bce_dice_adamw_cosine_s42
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import list_pairs, read_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", default="data/raw/Kvasir-SEG")
    ap.add_argument("--n-examples", type=int, default=6)
    args = ap.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    shutil.copy(Path(args.run) / "best.pt", "checkpoints/model.pt")

    test = set(read_split("data/splits", "kvasir", "test"))
    images = [p[1] for p in list_pairs(args.data_dir) if p[0] in test]
    examples = Path("app/examples")
    examples.mkdir(parents=True, exist_ok=True)
    for path in random.Random(0).sample(images, args.n_examples):
        shutil.copy(path, examples / path.name)
    print(f"checkpoints/model.pt <- {args.run}; {args.n_examples} test images copied to {examples}/")


if __name__ == "__main__":
    main()
