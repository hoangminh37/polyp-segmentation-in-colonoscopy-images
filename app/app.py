"""Gradio interface. Served together with the REST API by app/server.py:

    python -m app.server                 # http://127.0.0.1:7860  (UI at /, API at /api/v1, /docs)
    python app/app.py                    # same thing

The release folder defaults to checkpoints/ (override with POLYP_MODEL_DIR); build it with
scripts/package_models.py.
"""
import hashlib
import html
import json
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

if __package__ in (None, ""):  # run as a script (python app/app.py): make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from app import monitoring, render
from src.inference import ENSEMBLE, RawPrediction, finalize, overlap_scores, predict_raw

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"
CSS_PATH = APP_DIR / "style.css"
VIEWS = ["Overlay", "Probability", "Disagreement", "Errors vs ground truth"]
REGION_COLUMNS = ["#", "Area (% of image)", "Pixels", "Bounding box (x, y, w, h)", "Mean probability"]

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="*primary_700",
    button_primary_background_fill_hover="*primary_800",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover_dark="*primary_500",
    block_title_text_weight="500",
    block_label_text_weight="500",
)


# --- small helpers ----------------------------------------------------------------------------

def _fmt(value, digits=3, suffix=""):
    return "–" if value is None or value != value else f"{value:.{digits}f}{suffix}"


def _thumb(rgb):
    return cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


class Examples:
    """Example images and their ground-truth masks; recognises an example when it comes back
    from the browser (same size, near-identical thumbnail)."""

    def __init__(self, folder):
        index = folder / "examples.json"
        self.items = json.loads(index.read_text()) if index.exists() else []
        for item in self.items:
            item["path"] = str(folder / item["image"])
            rgb = np.array(Image.open(item["path"]).convert("RGB"))
            item["shape"], item["thumb"] = rgb.shape, _thumb(rgb)
            item["mask_path"] = str(folder / item["mask"])

    @property
    def paths(self):
        return [i["path"] for i in self.items]

    @property
    def gallery(self):
        return [(i["path"], i["label"]) for i in self.items]

    def load(self, index):
        item = self.items[index]
        note = f'<div class="example-note"><b>{html.escape(item["label"])}</b> · {html.escape(item.get("note", ""))}</div>'
        return np.array(Image.open(item["path"]).convert("RGB")), note

    def reference(self, rgb):
        for item in self.items:
            if item["shape"] == rgb.shape and np.abs(item["thumb"] - _thumb(rgb)).mean() < 1.5:
                mask = np.array(Image.open(item["mask_path"]).convert("L")) > 127
                return mask, item["source"]
        return None, None


class PredictionCache:
    """Recent model outputs, so changing the threshold or the view does not re-run the models."""

    def __init__(self, size=24):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.items[key] = value
            while len(self.items) > self.size:
                self.items.popitem(last=False)


def _truth(gt_upload, rgb, examples):
    """Ground truth for this image: an uploaded mask wins over a recognised example."""
    if gt_upload is not None:
        g = gt_upload if gt_upload.ndim == 2 else cv2.cvtColor(gt_upload[..., :3], cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return g > 127, "uploaded mask"
    return examples.reference(rgb)


# --- HTML fragments ---------------------------------------------------------------------------

def _tile(label, value, note=""):
    note = f'<span class="tile-note">{html.escape(note)}</span>' if note else ""
    return f'<div class="tile"><span class="tile-label">{label}</span><span class="tile-value">{value}</span>{note}</div>'


def stats_html(pred, scores=None, truth_source=None):
    tiles = [
        _tile("Polyp area", f"{pred.area_pct:.1f}%", "of the image"),
        _tile("Regions", str(len(pred.regions))),
        _tile("Model time", f"{pred.latency_ms:.0f} ms", f"{pred.passes} forward pass" + ("es" if pred.passes > 1 else "")),
        _tile("Peak probability", f"{pred.prob.max():.2f}"),
    ]
    out = f'<div class="tiles">{"".join(tiles)}</div>'
    if scores:
        gt = [_tile(k.capitalize() if k != "iou" else "IoU", f"{scores[k]:.3f}") for k in ("dice", "iou", "precision", "recall")]
        out += (f'<div class="section-label">Against ground truth <span>({html.escape(truth_source)}; '
                f'computed at the image\'s full resolution)</span></div>'
                f'<div class="tiles tiles-gt">{"".join(gt)}</div>')
    return out


EMPTY_STATS = '<div class="empty">Upload a colonoscopy frame or pick one of the examples.</div>'


def legend_html(view, threshold):
    c = render.LEGEND_COLOURS
    if view == "Overlay":
        return f'<div class="legend"><span class="swatch" style="background:{c["contour"]}"></span>Predicted polyp (probability &gt; {threshold:.2f})</div>'
    if view == "Probability":
        return ('<div class="legend"><span class="ramp ramp-viridis"></span><span class="ramp-ends">0</span>'
                '<span class="ramp-ends">1</span>&nbsp; polyp probability; white outline = threshold</div>')
    if view == "Disagreement":
        return (f'<div class="legend"><span class="ramp ramp-magma"></span><span class="ramp-ends">0</span>'
                f'<span class="ramp-ends">≥ {render.SPREAD_MAX}</span>&nbsp; standard deviation across the averaged predictions</div>')
    return ('<div class="legend">'
            f'<span class="swatch" style="background:{c["tp"]}"></span>correct'
            f'<span class="swatch" style="background:{c["fp"]}"></span>false positive'
            f'<span class="swatch" style="background:{c["fn"]}"></span>missed'
            '<span class="swatch outline"></span>ground-truth outline</div>')


def notice_html(messages, kind="warn"):
    if not messages:
        return ""
    items = "".join(f"<li>{html.escape(m)}</li>" for m in messages)
    return f'<div class="notice notice-{kind}"><ul>{items}</ul></div>'


def model_note(registry, model_id):
    meta = registry.meta
    if model_id == ENSEMBLE:
        m = meta.get("ensemble", {}).get("metrics", {})
        cards = [registry.cards[i] for i in registry.ensemble_members]
        bits = []
        if "kvasir_test" in m:
            bits.append(f"Kvasir test Dice {m['kvasir_test']['plain']['dice']:.3f} "
                        f"({m['kvasir_test']['tta']['dice']:.3f} with TTA)")
        if "clinicdb" in m:
            bits.append(f"ClinicDB {m['clinicdb']['plain']['dice']:.3f}")
        if all(c.get("cpu_ms") for c in cards):
            bits.append(f"≈{sum(c['cpu_ms'] for c in cards):.0f} ms on CPU")
        return '<div class="model-note">Mean probability of ' + str(len(cards)) + " models. " + " · ".join(bits) + "</div>"
    card = registry.cards[model_id]
    m = card.get("metrics", {})
    bits = []
    if "kvasir_test" in m:
        bits.append(f"Kvasir test Dice {m['kvasir_test']['dice']:.3f}")
    if "clinicdb" in m:
        bits.append(f"ClinicDB {m['clinicdb']['dice']:.3f}")
    bits.append(f"{card['params_m']} M parameters")
    if card.get("cpu_ms"):
        bits.append(f"≈{card['cpu_ms']:.0f} ms on CPU")
    return '<div class="model-note">' + " · ".join(bits) + "</div>"


def models_table_html(registry):
    meta = registry.meta

    def cell(metrics, dataset, key="dice"):
        return _fmt(metrics.get(dataset, {}).get(key))

    rows = []
    best = max((c["metrics"].get("kvasir_test", {}).get("dice", 0) for c in meta["models"]), default=0)
    for c in meta["models"]:
        m = c.get("metrics", {})
        top = ' class="best"' if m.get("kvasir_test", {}).get("dice") == best else ""
        enc = c["encoder"] or "—"
        rows.append(
            f"<tr{top}><td>{html.escape(c['label'])}</td><td>{enc}</td><td>{'ImageNet' if c['pretrained'] else 'none'}</td>"
            f"<td class='num'>{c['params_m']}</td><td class='num'>{_fmt(c.get('cpu_ms'), 0)}</td>"
            f"<td class='num'>{cell(m, 'kvasir_test')}</td><td class='num'>{cell(m, 'kvasir_test', 'iou')}</td>"
            f"<td class='num'>{cell(m, 'clinicdb')}</td><td class='mono'>{c['source_run']}</td></tr>")
    ens = meta.get("ensemble", {})
    members = [registry.cards[i] for i in registry.ensemble_members]
    for kind, label in (("plain", "Ensemble"), ("tta", "Ensemble + flip TTA")):
        m = {d: v[kind] for d, v in ens.get("metrics", {}).items()}
        if not m:
            continue
        cpu = sum(c["cpu_ms"] for c in members if c.get("cpu_ms")) * (4 if kind == "tta" else 1)
        rows.append(
            f"<tr class='ensemble'><td>{label}</td><td colspan='2'>{' + '.join(c['short'] for c in members)}</td>"
            f"<td class='num'>{sum(c['params_m'] for c in members):.1f}</td><td class='num'>{cpu:.0f}</td>"
            f"<td class='num'>{cell(m, 'kvasir_test')}</td><td class='num'>{cell(m, 'kvasir_test', 'iou')}</td>"
            f"<td class='num'>{cell(m, 'clinicdb')}</td><td>—</td></tr>")
    head = ("<tr><th>Model</th><th>Encoder</th><th>Pretraining</th><th class='num'>Params (M)</th>"
            "<th class='num'>CPU ms</th><th class='num'>Kvasir Dice</th><th class='num'>Kvasir IoU</th>"
            "<th class='num'>ClinicDB Dice</th><th>Source run</th></tr>")
    return f'<div class="table-wrap"><table class="models">{head}{"".join(rows)}</table></div>'


def header_html(meta):
    gate = " · release gate passed" if meta.get("gate", {}).get("passed") else ""
    return f"""
<div class="app-header">
  <div>
    <h1>Polyp segmentation in colonoscopy images</h1>
    <p>Outlines polyps in a colonoscopy frame. Four networks trained on Kvasir-SEG, from a U-Net written
    from scratch to a SegFormer transformer, plus their ensemble.</p>
  </div>
  <div class="release mono">release {html.escape(str(meta.get('version')))} · {html.escape(str(meta.get('git_commit')))}{gate}</div>
</div>
<div class="disclaimer">Course project for Deep Learning (2026.1). Not a medical device and not validated
for clinical use; do not use it to make decisions about patients.</div>"""


MODEL_CARD_MD = """
#### How the numbers were obtained
- **Training data.** Kvasir-SEG: 1,000 colonoscopy frames with expert polyp masks, split 800 / 100 / 100
  (train / validation / test). Near-duplicate frames are kept in the same split, so the test images
  were never seen in any form during training.
- **External test.** CVC-ClinicDB: 612 frames from a different hospital and camera, used only for
  testing. It shows how much accuracy is lost on images from another clinic.
- **Which checkpoint ships.** Each architecture was trained with three seeds. The release contains the
  seed with the best *validation* Dice, so the test numbers above were not used for the choice.
- **Metrics.** Dice and IoU per image at 352 × 352, threshold 0.5, averaged over images. Values from
  papers that evaluate at full resolution are not directly comparable.
- **Packaging.** Weights are stored in float16. Before the release was written, every file was
  re-evaluated on the test split and had to reproduce its recorded Dice (release gate).

#### Known limitations
- Every training image contains a polyp. The models have never seen a healthy colon and will often
  mark *something* on a normal frame.
- Single frames only: no use of the video context a real endoscopy system would have.
- Images that look unlike the training set (other modalities, heavy blur, non-endoscopic photos) are
  flagged by a simple colour-statistics check, which cannot catch every case.
"""

API_MD = """
The same models are available over HTTP. Interactive documentation: [`/docs`](/docs).

```bash
# health and release
curl http://localhost:7860/health
curl http://localhost:7860/api/v1/models

# JSON: area, regions with bounding boxes, timing, input warnings (+ mask as base64 PNG)
curl -F "file=@frame.jpg" "http://localhost:7860/api/v1/predict?model=segformer_b2&tta=true&include_mask=true"

# the binary mask as a PNG, same size as the input
curl -F "file=@frame.jpg" "http://localhost:7860/api/v1/predict/mask?model=ensemble" -o mask.png
```

```python
import requests

with open("frame.jpg", "rb") as f:
    r = requests.post("http://localhost:7860/api/v1/predict",
                      params={"model": "ensemble", "threshold": 0.5}, files={"file": f})
print(r.json()["polyp_area_pct"], r.json()["regions"])
```

Prometheus metrics (request counts, latency per model, predicted-area and confidence
distributions, flagged inputs) are exposed at [`/metrics`](/metrics).
"""


# --- the interface ----------------------------------------------------------------------------

def build_demo(registry):
    examples = Examples(EXAMPLES_DIR)
    cache = PredictionCache()
    monitoring.set_release(registry.meta)
    model_choices = [(registry.label(m), m) for m in registry.choices]

    def raw_for(rgb, model_id, tta):
        key = (hashlib.blake2b(rgb.tobytes(), digest_size=16).hexdigest(), rgb.shape, model_id, bool(tta))
        raw = cache.get(key)
        if raw is None:
            try:
                raw = predict_raw(registry, rgb, model_id, tta)
            except Exception:
                monitoring.record_error(model_id, "ui")
                raise
            monitoring.record(finalize(raw), "ui")
            cache.put(key, raw)
        return raw

    def analyse(image, gt_upload, model_id, tta, threshold, min_area, view):
        if image is None:
            return (None, "", EMPTY_STATS, pd.DataFrame(columns=REGION_COLUMNS),
                    gr.DownloadButton(value=None, interactive=False))
        rgb = np.ascontiguousarray(image[..., :3]) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pred = finalize(raw_for(rgb, model_id, tta), threshold, min_area)
        truth, source = _truth(gt_upload, rgb, examples)
        scores = overlap_scores(pred.mask, truth) if truth is not None else None

        notes, shown = [], view
        if view == "Disagreement" and pred.spread is None:
            notes.append("Disagreement needs several predictions to compare: turn on flip TTA or choose the ensemble.")
            shown = "Overlay"
        if view == "Errors vs ground truth" and truth is None:
            notes.append("No ground truth for this image. Use one of the examples or upload a mask under "
                         "“Ground-truth mask”.")
            shown = "Overlay"
        picture = {"Overlay": lambda: render.overlay(rgb, pred),
                   "Probability": lambda: render.probability(rgb, pred),
                   "Disagreement": lambda: render.disagreement(rgb, pred),
                   "Errors vs ground truth": lambda: render.errors(rgb, pred, truth)}[shown]()

        regions = pd.DataFrame(
            [[i, round(r.area_pct, 2), r.area_px, f"{r.bbox[0]}, {r.bbox[1]}, {r.bbox[2]}, {r.bbox[3]}",
              round(r.mean_prob, 3)] for i, r in enumerate(pred.regions, 1)], columns=REGION_COLUMNS)
        mask_file = Path(tempfile.mkdtemp(prefix="polyp-")) / f"mask_{model_id}_t{threshold:.2f}.png"
        cv2.imwrite(str(mask_file), pred.mask * 255)

        return (picture, legend_html(shown, threshold) + notice_html(pred.warnings) + notice_html(notes, "info"),
                stats_html(pred, scores, source), regions, gr.DownloadButton(value=str(mask_file), interactive=True))

    def pick_example(evt: gr.SelectData):
        return examples.load(evt.index)

    def pick_example_image(evt: gr.SelectData):
        return examples.load(evt.index)[0]

    def compare(image, gt_upload, tta):
        if image is None:
            return [], pd.DataFrame(), ""
        rgb = np.ascontiguousarray(image[..., :3]) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        truth, source = _truth(gt_upload, rgb, examples)
        gallery, rows, warnings = [], [], []
        if truth is not None:
            as_pred = RawPrediction(truth.astype(np.float32), None, 0.0, "ground truth", 0, [])
            gt_view = render.overlay(rgb, finalize(as_pred))
            gallery.append((gt_view, "Ground truth"))
        for model_id in registry.choices:
            pred = finalize(raw_for(rgb, model_id, tta), 0.5)
            warnings = pred.warnings
            s = overlap_scores(pred.mask, truth) if truth is not None else None
            view = render.errors(rgb, pred, truth) if truth is not None else render.overlay(rgb, pred)
            name = "Ensemble" if model_id == ENSEMBLE else registry.cards[model_id]["short"]
            caption = name + (f" · Dice {s['dice']:.2f}" if s else f" · {pred.latency_ms:.0f} ms")
            gallery.append((view, caption))
            row = {"Model": name, "Area (%)": round(pred.area_pct, 2),
                   "Regions": len(pred.regions), "Time (ms)": round(pred.latency_ms)}
            if s:
                row |= {"Dice": round(s["dice"], 3), "IoU": round(s["iou"], 3),
                        "Precision": round(s["precision"], 3), "Recall": round(s["recall"], 3)}
            rows.append(row)
        legend = legend_html("Errors vs ground truth", 0.5) if truth is not None else legend_html("Overlay", 0.5)
        return gallery, pd.DataFrame(rows), legend + notice_html(warnings)

    with gr.Blocks(title="Polyp segmentation") as demo:
        gr.HTML(header_html(registry.meta), elem_classes="full-width")

        with gr.Tab("Segment"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=320):
                    image = gr.Image(type="numpy", label="Colonoscopy frame", sources=["upload", "clipboard"],
                                     height=340)
                    example_note = gr.HTML()
                    if examples.items:
                        picker = gr.Gallery(examples.gallery, label="Examples from the test sets (ground truth known)",
                                            columns=3, rows=2, height=300, allow_preview=False, object_fit="cover",
                                            buttons=[], elem_classes="example-picker")
                        picker.select(pick_example, None, [image, example_note], show_progress="hidden")
                    model = gr.Dropdown(model_choices, value=registry.default, label="Model")
                    note = gr.HTML(model_note(registry, registry.default))
                    tta = gr.Checkbox(False, label="Flip test-time augmentation",
                                      info="Average the prediction over 4 flipped copies (about 4× slower).")
                    threshold = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Threshold",
                                          info="Pixels with a higher probability are marked as polyp. "
                                               "Lower = more sensitive, more false alarms.")
                    min_area = gr.Slider(0, 2, 0, step=0.1, label="Ignore regions smaller than (% of image)",
                                         info="0 keeps the raw model output, as in the reported metrics.")
                    with gr.Accordion("Ground-truth mask (optional)", open=False):
                        gt = gr.Image(type="numpy", image_mode="L", label="Binary mask, white = polyp",
                                      sources=["upload"], height=200)
                    run = gr.Button("Segment", variant="primary")

                with gr.Column(scale=7, min_width=360):
                    view = gr.Radio(VIEWS, value="Overlay", show_label=False, elem_classes="view-switch")
                    result = gr.Image(label="Result", interactive=False, height=520, format="png",
                                      buttons=["download", "fullscreen"])
                    legend = gr.HTML()
                    stats = gr.HTML(EMPTY_STATS)
                    regions = gr.Dataframe(headers=REGION_COLUMNS, label="Detected regions (largest first)",
                                           interactive=False, wrap=True, max_height=220)
                    download = gr.DownloadButton("Download mask (PNG)", interactive=False, size="sm")

            inputs = [image, gt, model, tta, threshold, min_area, view]
            outputs = [result, legend, stats, regions, download]
            run.click(analyse, inputs, outputs)
            image.change(analyse, inputs, outputs)
            image.upload(lambda: "", None, example_note, show_progress="hidden")
            gt.change(analyse, inputs, outputs, show_progress="hidden")
            for control in (model, tta, view):
                control.change(analyse, inputs, outputs, show_progress="minimal")
            for slider in (threshold, min_area):
                slider.release(analyse, inputs, outputs, show_progress="hidden")
            model.change(lambda m: model_note(registry, m), model, note, show_progress="hidden")

        with gr.Tab("Compare models"):
            gr.Markdown("Runs every model on the same frame (threshold 0.5). With ground truth, each result "
                        "is shown as an error map so the differences between models are easy to see.",
                        elem_classes="intro")
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300):
                    c_image = gr.Image(type="numpy", label="Colonoscopy frame", sources=["upload", "clipboard"],
                                       height=300)
                    if examples.items:
                        c_picker = gr.Gallery(examples.gallery, label="Examples", columns=3, rows=2, height=300,
                                              allow_preview=False, object_fit="cover", buttons=[],
                                              elem_classes="example-picker")
                        c_picker.select(pick_example_image, None, c_image, show_progress="hidden")
                    c_tta = gr.Checkbox(False, label="Flip test-time augmentation")
                    with gr.Accordion("Ground-truth mask (optional)", open=False):
                        c_gt = gr.Image(type="numpy", image_mode="L", label="Binary mask, white = polyp",
                                        sources=["upload"], height=200)
                    c_run = gr.Button("Run all models", variant="primary")
                with gr.Column(scale=8, min_width=360):
                    c_gallery = gr.Gallery(label="Results", columns=3, height="auto", object_fit="contain",
                                           preview=False, buttons=["download", "fullscreen"])
                    c_legend = gr.HTML()
                    c_table = gr.Dataframe(label="Summary", interactive=False)
            c_run.click(compare, [c_image, c_gt, c_tta], [c_gallery, c_table, c_legend])
            c_image.change(compare, [c_image, c_gt, c_tta], [c_gallery, c_table, c_legend])

        with gr.Tab("Models"):
            gr.HTML(models_table_html(registry), elem_classes="full-width")
            gr.Markdown(MODEL_CARD_MD, elem_classes="prose")

        with gr.Tab("API"):
            gr.Markdown(API_MD, elem_classes="prose")

        gr.HTML(elem_classes="full-width", value='<div class="footer">Data: Kvasir-SEG (Jha et al., MMM 2020) · CVC-ClinicDB (Bernal et al., '
                'CMIG 2015). Both datasets are licensed for research and education only.</div>')
    return demo


def css():
    return CSS_PATH.read_text()


if __name__ == "__main__":
    from app.server import main

    main()
