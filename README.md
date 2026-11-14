# Polyp Segmentation in Colonoscopy Images

Binary segmentation of polyps on [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) (1,000 images),
comparing three architectures, four losses, two optimizers and four LR schedules, with a Gradio demo.

| Approach | Model | Encoder | Params |
|---|---|---|---|
| Baseline CNN | U-Net | ResNet-34 | 24.4M |
| Advanced CNN | DeepLabV3+ (ASPP) | ResNet-34 | 22.4M |
| Transformer | SegFormer | MiT-B2 | 24.7M |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# dataset (46 MB, free for research/education, cite Jha et al. 2020)
mkdir -p data/raw && cd data/raw
curl -LO https://datasets.simula.no/downloads/kvasir-seg.zip && unzip -q kvasir-seg.zip && cd ../..
```

The train/val/test split (800/100/100) is committed in `data/splits/`, so everyone evaluates on
the same test images. It was produced by `python scripts/make_splits.py`, which keeps
near-duplicate frames (17 groups, 37 images) inside one split to avoid train/test leakage.

## Train

```bash
python -m src.train --config configs/unet_resnet34.yaml
python -m src.train --config configs/deeplabv3plus_resnet34.yaml
python -m src.train --config configs/segformer_mitb2.yaml

# any setting can be overridden
python -m src.train --config configs/unet_resnet34.yaml --loss focal --seed 1
python -m src.train --config configs/unet_resnet34.yaml --optimizer sgd --lr 0.01
python -m src.train --config configs/unet_resnet34.yaml --epochs 2 --name smoke   # quick check
```

Defaults (see `DEFAULTS` in `src/train.py`): 352×352 input, batch 16, 60 epochs, AdamW lr 1e-4,
cosine schedule, BCE + Dice loss, ImageNet-pretrained encoder. Add `--wandb` to log to Weights & Biases.

Each run writes to `runs/<name>/`: `best.pt` (best validation Dice), `log.csv`, `curves.png`,
`config.yaml`, and `metrics_kvasir_test.json` (the test split is evaluated automatically at the end).
The default run name encodes the setup: `<model>_<encoder>_<loss>_<optimizer>_<scheduler>_s<seed>`.

The whole grid, resumable:

```bash
PYTHON=.venv/bin/python bash scripts/run_experiments.sh models   # 3 models x 3 seeds
PYTHON=.venv/bin/python bash scripts/run_experiments.sh losses   # bce / dice / focal vs bce_dice
PYTHON=.venv/bin/python bash scripts/run_experiments.sh optim    # SGD, step / plateau / no schedule
```

### Where to train

The device is picked automatically (CUDA → Apple MPS → CPU). Measured on a MacBook M4 (16 GB),
batch 16 at 352×352:

| Model | Time per epoch | 60 epochs |
|---|---|---|
| U-Net / DeepLabV3+ (ResNet-34) | ~55 s | ~55 min |
| SegFormer MiT-B2 | ~2 min | ~2 h |

A laptop is fine for development and single runs. The full grid is 16 runs (~20 h on the M4),
so run it on a free Kaggle/Colab T4 GPU or split it between team members. `--img_size 256`
roughly halves the time at some cost on small polyps.

## Evaluate and report

```bash
python -m src.evaluate --run runs/<name>                      # Kvasir test split
python -m src.evaluate --run runs/<name> --data-dir data/raw/CVC-ClinicDB --name clinicdb
python -m src.visualize --runs runs/<name> --pick worst       # failure cases of one run
python scripts/summarize.py                                   # table, mean ± std over seeds

# comparisons across runs need a name and get their own folder
python -m src.plots runs/<unet> runs/<deeplab> runs/<segformer> --name models
python -m src.visualize --runs runs/<unet> runs/<deeplab> runs/<segformer> --name models --n 8
```

Where results live:

| Location | Content |
|---|---|
| `runs/<run>/` | everything about one run: config, log, curves, checkpoint, `metrics_<dataset>.json`, `per_image_<dataset>.csv`, `qualitative_<dataset>_<pick>.png` |
| `reports/<comparison>/` | figures comparing several runs, plus `runs.txt` listing exactly which runs were used |
| `reports/all_runs.csv`, `reports/summary.md` | one row per run and dataset (with the run name), and the aggregated table |

Metrics are Dice, IoU, precision and recall at threshold 0.5, computed **per image at the
model resolution and averaged over images**. Parameter count and latency (GPU and CPU) are reported too.

**Cross-dataset test.** To measure generalisation to another hospital/endoscope, put any other
polyp dataset (CVC-ClinicDB, CVC-ColonDB, ETIS-Larib) in a folder with `images/` and `masks/`
(matching file names) and pass it with `--data-dir`. The model is never trained on it.

## Demo

```bash
python scripts/package_demo.py --run runs/<best run>   # -> checkpoints/model.pt + app/examples/
python app/app.py                                      # http://127.0.0.1:7860

docker build -t polyp-demo . && docker run --rm -p 7860:7860 polyp-demo
```

The demo runs on CPU. Example images come from the held-out test split. It is a course project,
not a diagnostic tool.

## Known limitations

- Every Kvasir-SEG image contains a polyp, so the model has never seen a negative image and
  can produce false positives on healthy colon frames.
- With 100 test images, differences of about one Dice point between models are within noise;
  compare using mean ± std over seeds.
- Metrics are computed on resized 352×352 masks, not at the original resolution, so they are
  not directly comparable with papers that evaluate at full size.

## Layout

```
configs/    one YAML per architecture
src/        data.py  models.py  losses.py  metrics.py  train.py  evaluate.py  plots.py  visualize.py
scripts/    make_splits.py  run_experiments.sh  summarize.py  package_demo.py
app/        app.py (Gradio)
data/       splits/ (committed)   raw/, cache/ (ignored)
```
