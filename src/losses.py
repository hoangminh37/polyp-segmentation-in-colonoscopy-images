"""Loss functions for binary segmentation. All take raw logits and {0,1} float targets [B,1,H,W]."""
import torch
import torch.nn.functional as F


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def dice_loss(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce)  # probability assigned to the true class
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def bce_dice_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


LOSSES = {
    "bce": bce_loss,
    "dice": dice_loss,
    "focal": focal_loss,
    "bce_dice": bce_dice_loss,
}
