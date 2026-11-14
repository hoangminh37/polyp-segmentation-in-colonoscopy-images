#!/usr/bin/env bash
# Full experiment grid. Usage:
#   bash scripts/run_experiments.sh models      # 3 architectures x 3 seeds
#   bash scripts/run_experiments.sh losses      # loss ablation on U-Net
#   bash scripts/run_experiments.sh optim       # optimizer / scheduler ablation on U-Net
#   bash scripts/run_experiments.sh all
# Extra arguments are forwarded to every run, e.g. `... models --epochs 30`.
# A run whose test metrics already exist is skipped, so the script can be resumed.
set -euo pipefail
stage="${1:-all}"; shift || true
PY="${PYTHON:-python}"
UNET=configs/unet_resnet34.yaml

run() {  # run <config> <name> [overrides...]
  local config="$1" name="$2"; shift 2
  if [ -f "runs/$name/metrics_kvasir_test.json" ]; then echo "skip $name"; return; fi
  "$PY" -m src.train --config "$config" --name "$name" "$@"
}

if [[ "$stage" == models || "$stage" == all ]]; then
  for seed in 42 1 2; do
    run configs/unet_resnet34.yaml          "unet_resnet34_bce_dice_adamw_cosine_s$seed"          --seed "$seed" "$@"
    run configs/deeplabv3plus_resnet34.yaml "deeplabv3plus_resnet34_bce_dice_adamw_cosine_s$seed" --seed "$seed" "$@"
    run configs/segformer_mitb2.yaml        "segformer_mit_b2_bce_dice_adamw_cosine_s$seed"       --seed "$seed" "$@"
  done
fi

if [[ "$stage" == losses || "$stage" == all ]]; then
  for loss in bce dice focal; do   # bce_dice is the baseline from the `models` stage
    run $UNET "unet_resnet34_${loss}_adamw_cosine_s42" --loss "$loss" "$@"
  done
fi

if [[ "$stage" == optim || "$stage" == all ]]; then
  run $UNET unet_resnet34_bce_dice_sgd_cosine_s42    --optimizer sgd --lr 0.01 "$@"
  run $UNET unet_resnet34_bce_dice_adamw_step_s42    --scheduler step "$@"
  run $UNET unet_resnet34_bce_dice_adamw_plateau_s42 --scheduler plateau "$@"
  run $UNET unet_resnet34_bce_dice_adamw_none_s42    --scheduler none "$@"
fi

"$PY" scripts/summarize.py
