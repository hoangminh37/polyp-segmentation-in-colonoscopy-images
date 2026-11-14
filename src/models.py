"""Model factory for the three compared architectures."""
import segmentation_models_pytorch as smp
import torch

ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "segformer": smp.Segformer,
}


def build_model(model, encoder, encoder_weights="imagenet"):
    return ARCHS[model](encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)


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
    cfg = ckpt["cfg"]
    model = build_model(cfg["model"], cfg["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters())
