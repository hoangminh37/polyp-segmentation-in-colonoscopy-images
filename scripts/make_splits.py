"""Create a fixed train/val/test split of Kvasir-SEG.

Near-duplicate frames (same polyp, almost the same view) are grouped with a
perceptual hash and kept in the same split, so the test set does not leak
into training.

    python scripts/make_splits.py
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import list_pairs  # noqa: E402


def dhash(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def group_duplicates(hashes, max_dist):
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    for i, j in zip(*np.where(np.triu(dist <= max_dist, k=1))):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(hashes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/Kvasir-SEG")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--name", default="kvasir")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--max-hash-dist", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = list_pairs(args.data_dir)
    names = [p[0] for p in pairs]
    hashes = np.stack([dhash(p[1]) for p in pairs])
    groups = group_duplicates(hashes, args.max_hash_dist)
    multi = [g for g in groups if len(g) > 1]
    print(f"{len(names)} images, {len(groups)} groups; "
          f"{len(multi)} near-duplicate groups covering {sum(map(len, multi))} images")

    random.Random(args.seed).shuffle(groups)
    n_test = round(len(names) * args.test_frac)
    n_val = round(len(names) * args.val_frac)
    splits = {"test": [], "val": [], "train": []}
    for g in groups:
        if len(splits["test"]) < n_test:
            splits["test"] += g
        elif len(splits["val"]) < n_val:
            splits["val"] += g
        else:
            splits["train"] += g

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, idx in splits.items():
        (out / f"{args.name}_{split}.txt").write_text("\n".join(sorted(names[i] for i in idx)) + "\n")
        print(f"{split}: {len(idx)}")


if __name__ == "__main__":
    main()
