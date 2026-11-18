# Polyp Segmentation in Colonoscopy Images

Binary segmentation of polyps on [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) (1,000 images).
The project climbs a ladder from a U-Net written from scratch to transfer learning and advanced
architectures, ablates architecture choices, losses, optimizers and LR schedules, tests
generalisation on an external hospital's images, and ships the models as a Gradio demo and REST API in Docker, with
experiment tracking, monitoring, tests and CI.

| Approach | Model | Encoder | Params |
|---|---|---|---|
| Course baseline | Custom U-Net, written in `src/custom_unet.py` | none (trained from scratch) | 7.8M |
| Transfer-learning check | U-Net | ResNet-34, no pretraining | 24.4M |
| Baseline CNN | U-Net | ResNet-34, ImageNet | 24.4M |
| Advanced CNN | DeepLabV3+ (ASPP) | ResNet-34, ImageNet | 22.4M |
| Transformer | SegFormer | MiT-B2, ImageNet | 24.7M |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt     # tests, lint, load test (optional)

# training data: Kvasir-SEG (46 MB, free for research/education, cite Jha et al. 2020)
mkdir -p data/raw && cd data/raw
curl -LO https://datasets.simula.no/downloads/kvasir-seg.zip && unzip -q kvasir-seg.zip && cd ../..
```

The train/val/test split (800/100/100) is committed in `data/splits/`, so everyone evaluates on
the same test images. It was produced by `python scripts/make_splits.py`, which keeps
near-duplicate frames (17 groups, 37 images) inside one split to avoid train/test leakage.

**External test set (evaluation only).** CVC-ClinicDB (612 frames from Hospital Clinic, Barcelona;
non-commercial use only, cite Bernal et al. 2015) measures how well models trained on Kvasir-SEG
transfer to another hospital. It is never used for training. Mirror used here:
[huggingface.co/datasets/berkaytrhn/cvc-clinicdb](https://huggingface.co/datasets/berkaytrhn/cvc-clinicdb).

```bash
mkdir -p data/raw/CVC-ClinicDB/images data/raw/CVC-ClinicDB/masks && cd data/raw/CVC-ClinicDB
base=https://huggingface.co/datasets/berkaytrhn/cvc-clinicdb/resolve/main/CVC-ClinicDB
for i in $(seq 1 612); do
  curl -sL "$base/Original/$i.png" -o images/$i.png
  curl -sL "$base/Ground%20Truth/$i.png" -o masks/$i.png
done; cd ../../..
```

## Train one run

```bash
python -m src.train --config configs/custom_unet.yaml
python -m src.train --config configs/unet_resnet34.yaml
python -m src.train --config configs/unet_resnet34.yaml --loss tversky --seed 1
python -m src.train --config configs/custom_unet.yaml --no-skip --norm group
python -m src.train --config configs/unet_resnet34.yaml --epochs 2 --name smoke   # quick check
```

Every option is listed with its meaning in `DEFAULTS` in `src/config.py`. The defaults are 352×352
input, batch 16, 40 epochs, AdamW (lr 1e-4, weight decay 1e-2), cosine schedule, BCE + Dice loss
and an ImageNet-pretrained encoder. Add `--mlflow` to log the run to MLflow (see below) or
`--wandb` for Weights & Biases.

| Option | Choices |
|---|---|
| `--loss` | `bce`, `wbce` (weighted, `--pos_weight`), `focal`, `dice`, `tversky` (`--tversky_alpha/beta`), `bce_dice`, `dice_focal` |
| `--optimizer` | `adamw`, `adam`, `sgd` (momentum 0.9, Nesterov) |
| `--scheduler` | `cosine`, `step`, `plateau`, `none`; plus `--warmup_epochs N` |
| Custom U-Net | `--base_channels`, `--skip/--no-skip`, `--upsample transpose/bilinear`, `--norm batch/group/none`, `--block plain/residual`, `--dropout` |

Each run writes everything about itself to `runs/<name>/`: `config.yaml`, `log.csv` (one row
per epoch), `curves.png`, `best.pt` (checkpoint with the best validation Dice), and for every test
set a `metrics_<dataset>.json` and a `per_image_<dataset>.csv`.

## Run the experiment grid

All runs are defined in `scripts/experiments.py`, in four groups:

| Group | What changes | Runs |
|---|---|---|
| `main` | the five models above | 5 models × seeds 42, 1, 2 |
| `arch` | custom U-Net: no skip, bilinear upsampling, GroupNorm, no norm, residual blocks, dropout | 6 + baseline, seed 42 |
| `loss` | U-Net R34: BCE, weighted BCE, Focal, Dice, Tversky, Dice + Focal | 6 + baseline, seed 42 |
| `optim` | U-Net R34: Adam, SGD, SGD + warm-up, step, plateau, constant LR | 6 + baseline, seed 42 |

```bash
python scripts/experiments.py --list                        # status of every run
python scripts/experiments.py --group main --seeds 42       # train what is missing
python scripts/experiments.py                               # everything (~24 h on an M4)
```

The runner skips finished runs, so it can be stopped and restarted, and evaluates every run on
CVC-ClinicDB when that folder exists. The device is picked automatically (CUDA → Apple MPS → CPU).
Measured on a MacBook M4 (16 GB), one 40-epoch run takes about 35 min for U-Net/DeepLabV3+,
50–60 min for the custom U-Net and 85 min for SegFormer. To share the load, members can run
different groups on Kaggle/Colab and copy the `runs/<name>/` folders back.

## Report

```bash
python -m src.ensemble --runs runs/unet_r34_s42 runs/deeplabv3plus_r34_s42 runs/segformer_b2_s42
python scripts/make_report.py
```

`src.ensemble` measures test-time augmentation (average over flips) and the ensemble of models,
without training. `make_report.py` reads every finished run and writes:

| Location | Content |
|---|---|
| `reports/summary.md` | all result tables (mean ± std over seeds) with links to the figures |
| `reports/all_runs.csv` | one row per run and test set |
| `reports/main/` | learning curves, test metrics, in-domain vs ClinicDB Dice, Dice by polyp size, accuracy vs inference time, qualitative grids |
| `reports/arch/`, `reports/loss/`, `reports/optim/` | test metrics, and validation curves of each variant against the group baseline |
| `reports/ensemble/` | TTA and ensemble results |

Metrics are Dice, IoU, precision and recall at threshold 0.5, computed **per image at 352×352 and
averaged over images**. Single-run tools: `python -m src.evaluate --run runs/<name> [--data-dir ...]`,
`python -m src.visualize --runs runs/<name> --pick worst`, `python -m src.plots runs/<name>`.

## Demo and API

```bash
python scripts/package_models.py      # build the release: checkpoints/ + app/examples/   (~5 min)
python -m app.server                  # http://127.0.0.1:7860
```

`package_models.py` turns finished runs into a **release**: for each architecture it takes the seed
with the best *validation* Dice, stores the weights in float16 with a SHA-256 checksum, and writes
`checkpoints/registry.json` with each model's test metrics, latency and source run. Before writing,
a **release gate** re-evaluates every stored file on the Kvasir test split and stops if its Dice
differs from the recorded one by more than 0.005 (measured: ≤ 0.0001 after the float16 conversion).
It also measures the ensemble, records colour statistics of the training images for the input
check, and picks example images (small, typical, large, two polyps, a failure case, ClinicDB) with
their ground-truth masks.

The interface (Gradio, `app/app.py`):

| Tab | What it shows |
|---|---|
| Segment | one model or the ensemble on a frame; views: overlay, probability map, disagreement between averaged predictions (ensemble / flip TTA), error map against the ground truth; area, regions with bounding boxes, timing; Dice/IoU when the ground truth is known (examples, or upload a mask); mask download |
| Compare models | every model on the same frame, side by side, with error maps and a summary table |
| Models | test metrics of the release (Kvasir-SEG and CVC-ClinicDB), how they were obtained, limitations |
| API | how to call the REST API |

Images whose colour statistics are far from the training images (more than 4 standard deviations on
any of mean R/G/B, saturation, contrast) get a warning: 0 of 100 Kvasir test images and 5 of 612
ClinicDB frames are flagged, every non-endoscopic image we tried is.

The same process serves a REST API (`app/server.py`, OpenAPI docs at `/docs`):

```bash
curl http://localhost:7860/health
curl -F "file=@frame.jpg" "http://localhost:7860/api/v1/predict?model=ensemble&tta=true"   # JSON
curl -F "file=@frame.jpg" "http://localhost:7860/api/v1/predict/mask" -o mask.png          # PNG mask
```

Environment variables: `POLYP_MODEL_DIR` (release folder), `WARMUP` (`default`/`all`/`none`),
`MAX_UPLOAD_MB`, `PREDICTION_LOG` (JSONL file, one summary line per prediction, never the image),
`HOST`, `PORT`.

## Docker and MLOps

```bash
docker build -t polyp-seg . && docker run --rm -p 7860:7860 polyp-seg    # CPU, non-root, with healthcheck
docker compose up -d --build                         # same, logs predictions to ./logs/
docker compose --profile monitoring up -d --build    # + Prometheus :9090 and Grafana :3000
```

The image contains only `src/`, `app/` and the release (no datasets, no runs) with dependencies
pinned in `requirements-serve.txt`. `make help` lists the shortcuts.

| Concern | How it is handled |
|---|---|
| Experiment tracking | Every run writes `runs/<name>/` (config, per-epoch log, checkpoint, metrics). `--mlflow` logs live to MLflow; `scripts/log_to_mlflow.py` imports finished runs from any machine (idempotent, tagged by experiment group) and records the release. `make mlflow-ui` to browse. |
| Reproducibility | Fixed seeds (incl. the augmentation RNG), committed data splits, the config stored in each run and checkpoint, git commit in the release. |
| Model packaging | `scripts/package_models.py`: selection on validation Dice, float16 weights, checksums verified at load time, release gate. |
| Serving | FastAPI + Gradio in one process; input validation (type, size, decodability); models loaded once and warmed up. |
| Monitoring | `/metrics` for Prometheus: requests, errors, model time per model, and two *output* distributions (predicted area, peak probability) plus the share of flagged inputs, which reveal input drift without labels. Grafana dashboard provisioned from `deploy/`. |
| Load / drift test | `scripts/load_test.py <image folder>` reports throughput and latency percentiles; `--drift` sends non-endoscopic images so the drift panels react. |
| Tests and CI | `pytest` (35 tests: metrics, losses, model shapes, inference, checksums, API) runs without the dataset on a tiny random release. GitHub Actions runs lint and tests, then builds the image, starts it and checks `/health` and a prediction. |

Measured in the container on a MacBook M4 (CPU, ClinicDB frames, 2 concurrent clients):
SegFormer about 4.8 requests/s, p95 latency 0.53 s.

## Known limitations

- Every Kvasir-SEG image contains a polyp, so the model has never seen a negative image and
  can produce false positives on healthy colon frames.
- With 100 test images, differences of about one Dice point between models are within noise;
  compare using mean ± std over seeds.
- Ablations use one seed, so small differences there are not conclusive.
- Metrics are computed on resized 352×352 masks, not at the original resolution, so they are
  not directly comparable with papers that evaluate at full size.

## Layout

```
configs/    one YAML per model
src/        config.py data.py custom_unet.py models.py losses.py metrics.py train.py
            evaluate.py ensemble.py plots.py visualize.py tracking.py inference.py
scripts/    make_splits.py experiments.py make_report.py
            package_models.py log_to_mlflow.py load_test.py
app/        app.py (Gradio UI) server.py (FastAPI) render.py monitoring.py style.css
tests/      pytest suite (no dataset needed)
deploy/     prometheus.yml, Grafana provisioning and dashboard
data/       splits/ (committed)   raw/, cache/ (ignored)
checkpoints/  model release (ignored, built by package_models.py)
```
