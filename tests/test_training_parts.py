"""Metrics, losses and model factory."""
import pytest
import torch

from src.config import with_defaults
from src.losses import LOSSES, build_loss
from src.metrics import batch_metrics
from src.models import build_model, count_params


def test_perfect_prediction_scores_one():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    logits = (target * 2 - 1) * 20
    m = batch_metrics(logits, target)
    for name in ("dice", "iou", "precision", "recall"):
        assert torch.allclose(m[name], torch.ones(2), atol=1e-5), name


def test_disjoint_prediction_scores_zero():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1
    logits = torch.full_like(target, -20)
    logits[..., 4:] = 20
    assert batch_metrics(logits, target)["dice"].item() < 1e-5


@pytest.mark.parametrize("name", LOSSES)
def test_losses_are_finite_and_prefer_the_right_answer(name):
    loss_fn = build_loss(with_defaults({"loss": name}))
    target = (torch.rand(2, 1, 16, 16) > 0.7).float()
    good, bad = (target * 2 - 1) * 5, (1 - target * 2) * 5
    assert torch.isfinite(loss_fn(good, target))
    assert loss_fn(good, target) < loss_fn(bad, target)


@pytest.mark.parametrize("cfg", [
    {"model": "custom_unet", "base_channels": 8},
    {"model": "custom_unet", "base_channels": 8, "skip": False, "upsample": "bilinear", "norm": "group",
     "block": "residual", "dropout": 0.2},
    {"model": "unet", "encoder": "resnet18"},
    {"model": "deeplabv3plus", "encoder": "resnet18"},
    {"model": "segformer", "encoder": "mit_b0"},
])
def test_models_output_one_logit_per_pixel(cfg):
    model = build_model(with_defaults(cfg), pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert count_params(model) > 0
