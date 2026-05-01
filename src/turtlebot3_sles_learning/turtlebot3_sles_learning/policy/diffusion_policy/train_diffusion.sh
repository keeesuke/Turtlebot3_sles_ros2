#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_NPZ="dataset/train_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz"
VAL_NPZ="dataset/val_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz"
SAVE_DIR="models/diffusion_policy"
RUN_NAME="diffusion_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$SAVE_DIR"
LOG="$SAVE_DIR/${RUN_NAME}.log"

echo "Logging to $LOG"

python train_diffusion_policy.py \
  --train-npz "$TRAIN_NPZ" \
  --val-npz   "$VAL_NPZ" \
  --save-dir  "$SAVE_DIR" \
  --epochs 50 \
  --batch-size 2048 \
  --lr 1e-4 \
  --num-train-timesteps 100 \
  --num-inference-steps 100 \
  --num-eval-inference-steps 10 \
  --num-eval-samples 8 \
  --val-max-batches 20 \
  --full-eval-every 5 \
  --plot-every 5 \
  --wandb-project diffusion-policy \
  --wandb-run-name "$RUN_NAME" \
  --device cuda \
  2>&1 | tee "$LOG"
