#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="$HOME/robot_data/real_world_datasets/diffusion_h50"
TRAIN_NPZ="$DATA_DIR/train_dataset_h50_hist10.npz"
VAL_NPZ="$DATA_DIR/val_dataset_h50_hist10.npz"
SAVE_DIR="$HOME/robot_data/real_world_models/flow_h50"
RUN_NAME="flow_real_$(date +%Y%m%d_%H%M%S)"
PYTHON="/home/acrl/miniforge3/envs/precision_manip/bin/python3"

# Collision penalty weight — set to 0.0 to disable.
# Good starting values: 0.1 – 1.0.  Robot radius for TurtleBot3 Burger ≈ 0.105 m.
COLLISION_WEIGHT="${COLLISION_WEIGHT:-0.5}"
ROBOT_RADIUS="${ROBOT_RADIUS:-0.15}"

mkdir -p "$SAVE_DIR"
LOG="$SAVE_DIR/${RUN_NAME}.log"

echo "Run: $RUN_NAME"
echo "Logging to $LOG"
echo "Collision penalty weight: $COLLISION_WEIGHT  robot_radius: $ROBOT_RADIUS"

$PYTHON train_flow_matching.py \
  --train-npz "$TRAIN_NPZ" \
  --val-npz   "$VAL_NPZ" \
  --save-dir  "$SAVE_DIR" \
  --epochs 1000 \
  --batch-size 512 \
  --lr 1e-4 \
  --num-inference-steps 100 \
  --num-eval-inference-steps 10 \
  --num-eval-samples 8 \
  --val-every 10 \
  --val-max-batches 20 \
  --full-eval-every 10 \
  --integrator euler \
  --plot-every 10 \
  --device cuda \
  --collision-loss-weight "$COLLISION_WEIGHT" \
  --robot-radius "$ROBOT_RADIUS" \
  2>&1 | tee "$LOG"
