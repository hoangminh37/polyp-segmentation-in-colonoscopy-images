"""Model factory: the from-scratch baseline and the three library architectures."""
import segmentation_models_pytorch as smp
import torch

from .config import with_defaults
from .custom_unet import CustomUNet

ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "segformer": smp.Segformer,
}


def build_model(cfg, pretrained=True):
    """Build the model described by a run config. pretrained=False skips downloading weights
    (used when a checkpoint is loaded right after)."""
    cfg = with_defaults(cfg)
    if cfg["model"] == "custom_unet":
        return CustomUNet(cfg["base_channels"], cfg["skip"], cfg["upsample"], cfg["norm"],
                          cfg["block"], cfg["dropout"])
    weights = cfg["encoder_weights"]
    if not pretrained or weights in (None, "", "none"):
        weights = None
    return ARCHS[cfg["model"]](encoder_name=cfg["encoder"], encoder_weights=weights, in_channels=3, classes=1)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device="cpu"):
    """Rebuild a trained model from a checkpoint. Returns (model, cfg)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = with_defaults(ckpt["cfg"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters())
