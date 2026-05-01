#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="$HOME/robot_data/real_world_datasets/diffusion_h50"
TRAIN_NPZ="$DATA_DIR/train_dataset_h50_hist10.npz"
VAL_NPZ="$DATA_DIR/val_dataset_h50_hist10.npz"
SAVE_DIR="$HOME/robot_data/real_world_models/diffusion_h50"
RUN_NAME="diffusion_real_$(date +%Y%m%d_%H%M%S)"
PYTHON="/home/acrl/miniforge3/envs/precision_manip/bin/python3"

mkdir -p "$SAVE_DIR"
LOG="$SAVE_DIR/${RUN_NAME}.log"

echo "Run: $RUN_NAME"
echo "Logging to $LOG"

$PYTHON train_diffusion_policy.py \
  --train-npz "$TRAIN_NPZ" \
  --val-npz   "$VAL_NPZ" \
  --save-dir  "$SAVE_DIR" \
  --epochs 100 \
  --batch-size 512 \
  --lr 1e-4 \
  --num-train-timesteps 100 \
  --num-inference-steps 100 \
  --num-eval-inference-steps 10 \
  --num-eval-samples 8 \
  --val-max-batches 20 \
  --full-eval-every 5 \
  --plot-every 5 \
  --device cuda \
  2>&1 | tee "$LOG"
