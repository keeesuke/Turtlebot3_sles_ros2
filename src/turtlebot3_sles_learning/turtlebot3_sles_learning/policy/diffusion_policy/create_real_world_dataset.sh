#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="$HOME/robot_data"
OUT_DIR="$HOME/robot_data/real_world_datasets/diffusion"

# 10 Hz to match MPC frequency; horizon=20 → 2.0 s lookahead
# Kanayama controller (50 Hz) tracks the predicted waypoints
python3 create_real_world_dataset.py \
  --data-root "$DATA_ROOT" \
  --out-dir   "$OUT_DIR" \
  --include-augmented \
  --dt 0.1 --horizon 20 \
  --history-len 10 \
  --v-max 0.26 \
  --w-max 1.82 \
  --lidar-max-range 3.5 \
  --train-frac 0.80 \
  --val-frac   0.10 \
  --seed 42
