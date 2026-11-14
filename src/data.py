"""Dataset loading, caching and augmentation.

Raw colonoscopy images are large (up to 1920x1072) and decoding them every epoch
is the main bottleneck on a laptop. Each dataset is therefore resized once to
`img_size` x `img_size` and cached as a single .npz that is loaded into RAM.
"""
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def list_pairs(root):
    """Return sorted [(stem, image_path, mask_path)] for a folder with images/ and masks/."""
    root = Path(root)
    images = {p.stem: p for p in (root / "images").iterdir() if p.suffix.lower() in IMG_EXTS}
    masks = {p.stem: p for p in (root / "masks").iterdir() if p.suffix.lower() in IMG_EXTS}
    stems = sorted(images.keys() & masks.keys())
    if not stems:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")
    return [(s, images[s], masks[s]) for s in stems]


def load_arrays(root, img_size, cache_dir="data/cache"):
    """Load a dataset as (names, images[N,H,W,3] uint8 RGB, masks[N,H,W] uint8 in {0,1})."""
    root = Path(root)
    cache = Path(cache_dir) / f"{root.name}_{img_size}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return list(d["names"]), d["images"], d["masks"]

    pairs = list_pairs(root)
    print(f"Caching {len(pairs)} images from {root} at {img_size}x{img_size} -> {cache}")
    images = np.zeros((len(pairs), img_size, img_size, 3), np.uint8)
    masks = np.zeros((len(pairs), img_size, img_size), np.uint8)
    for i, (_, img_path, mask_path) in enumerate(pairs):
        img = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        images[i] = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        # Kvasir-SEG masks are JPEGs with compression noise, so binarise after resizing.
        masks[i] = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_AREA) > 127
    names = [p[0] for p in pairs]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(names), images=images, masks=masks)
    return names, images, masks


def read_split(split_dir, dataset_name, split):
    path = Path(split_dir) / f"{dataset_name}_{split}.txt"
    return path.read_text().split()


def build_transforms(train, seed=None):
    if not train:
        return A.Compose([A.Normalize(MEAN, STD), ToTensorV2()])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.06, 0.06), rotate=(-20, 20), p=0.5),
            A.ElasticTransform(alpha=60, sigma=8, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ],
        seed=seed,  # albumentations has its own RNG; torch/numpy seeds do not reach it
    )


class PolypDataset(Dataset):
    def __init__(self, names, images, masks, subset=None, train=False, seed=None):
        if subset is not None:
            index = {n: i for i, n in enumerate(names)}
            idx = [index[n] for n in subset]
            names = [names[i] for i in idx]
            images, masks = images[idx], masks[idx]
        self.names, self.images, self.masks = names, images, masks
        self.transform = build_transforms(train, seed)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        out = self.transform(image=self.images[i], mask=self.masks[i])
        return out["image"], out["mask"].unsqueeze(0).float()


def preprocess_image(rgb, img_size):
    """Single RGB uint8 image -> normalised tensor [1,3,S,S] (used by the demo app)."""
    img = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
