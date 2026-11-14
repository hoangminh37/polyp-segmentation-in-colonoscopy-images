"""Segmentation metrics, computed per image and then averaged over images."""
import torch

METRICS = ("dice", "iou", "precision", "recall")


def batch_metrics(logits, target, threshold=0.5, eps=1e-7):
    """Return {metric: tensor[B]} for one batch."""
    pred = (torch.sigmoid(logits) > threshold).float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }
