"""Gradio demo: upload a colonoscopy image, get the predicted polyp mask and overlay.

    POLYP_CKPT=runs/<run>/best.pt python app/app.py      # default: checkpoints/model.pt
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import gradio as gr
import numpy as np
import torch

from src.data import preprocess_image
from src.models import load_checkpoint

CKPT = os.environ.get("POLYP_CKPT", str(ROOT / "checkpoints" / "model.pt"))
EXAMPLES_DIR = Path(__file__).parent / "examples"
DEVICE = torch.device("cpu")  # CPU keeps the demo portable (Docker, laptops without GPU)

model, cfg = load_checkpoint(CKPT, DEVICE)


@torch.no_grad()
def segment(image, threshold):
    if image is None:
        return None, None, "Upload an image first."
    start = time.perf_counter()
    x = preprocess_image(image, cfg["img_size"])
    prob = torch.sigmoid(model(x))[0, 0].numpy()
    latency = (time.perf_counter() - start) * 1000

    h, w = image.shape[:2]
    mask = (cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) > threshold).astype(np.uint8)
    overlay = image.copy()
    overlay[mask == 1] = (0.6 * overlay[mask == 1] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), max(2, w // 300))

    info = (f"Predicted polyp area: {mask.mean() * 100:.1f}% of the image\n"
            f"Regions found: {len(contours)}\n"
            f"Inference time: {latency:.0f} ms (CPU)\n"
            f"Model: {cfg['model']} / {cfg['encoder']} at {cfg['img_size']}x{cfg['img_size']}")
    return overlay, mask * 255, info


examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

with gr.Blocks(title="Polyp Segmentation") as demo:
    gr.Markdown(
        "# Polyp Segmentation in Colonoscopy Images\n"
        "Course project demo. **Not a medical device — do not use for diagnosis.** "
        "The model was trained only on images that contain a polyp, so it can mark a region "
        "even on a healthy colon image."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy", label="Colonoscopy image")
            threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Threshold")
            button = gr.Button("Segment", variant="primary")
        with gr.Column():
            overlay = gr.Image(label="Overlay")
            mask = gr.Image(label="Binary mask")
            info = gr.Textbox(label="Details", lines=4)
    if examples:
        gr.Examples(examples, inputs=image, label="Examples (held-out test images)")
    button.click(segment, [image, threshold], [overlay, mask, info])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)))
