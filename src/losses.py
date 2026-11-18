"""Loss functions for binary segmentation (L10). All take raw logits and {0,1} targets [B,1,H,W].

Distribution-based: bce, wbce (weighted BCE), focal
Region-based:       dice, tversky
Combined:           bce_dice, dice_focal
"""
from functools import partial

import torch
import torch.nn.functional as F


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def weighted_bce_loss(logits, target, pos_weight=2.0):
    """BCE with polyp pixels weighted by pos_weight (β in L10): β > 1 penalises false negatives more."""
    weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)


def dice_loss(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, smooth=1.0):
    """1 - TP / (TP + α·FP + β·FN). α = β = 0.5 gives Dice loss; β > α favours recall."""
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    return (1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce)  # probability assigned to the true class
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def bce_dice_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


def dice_focal_loss(logits, target):
    return dice_loss(logits, target) + focal_loss(logits, target)


LOSSES = ["bce", "wbce", "focal", "dice", "tversky", "bce_dice", "dice_focal"]


def build_loss(cfg):
    name = cfg["loss"]
    if name == "bce":
        return bce_loss
    if name == "wbce":
        return partial(weighted_bce_loss, pos_weight=cfg["pos_weight"])
    if name == "focal":
        return focal_loss
    if name == "dice":
        return dice_loss
    if name == "tversky":
        return partial(tversky_loss, alpha=cfg["tversky_alpha"], beta=cfg["tversky_beta"])
    if name == "bce_dice":
        return bce_dice_loss
    if name == "dice_focal":
        return dice_focal_loss
    raise ValueError(f"Unknown loss {name}; choose from {LOSSES}")
