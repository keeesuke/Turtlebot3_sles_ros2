#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python create_dataset_imitation_local_traj.py \
  --data-root dataset/dataset \
  --out-dir   dataset/ \
  --train-folders 740 \
  --val-folders   92 \
  --test-folders  91
