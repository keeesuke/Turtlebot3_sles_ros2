# Deterministic Local Trajectory Baseline

This is a **deterministic (non-diffusion)** baseline for local trajectory prediction, intended to replace/benchmark the current DDPM model.

It learns the supervised mapping:

\[
(\text{current state} + \text{history} + \text{lidar} + \text{target}) \rightarrow \text{future trajectory (H=20)}
\]

## What it predicts

Per horizon step \(k \in \{1..H\}\), the model predicts (in the **robot frame**):

- \((dx, dy, \sin(d\theta), \cos(d\theta))\)

The dataset stores outputs flattened as `(H*4)`, but the model **operates on `(B, H, 4)`** for temporal modeling (per your requirement).

## Inputs / dataset format

The training script uses the same `.npz` dataset format as the diffusion code:

- `inputs`: `(N, input_dim)` float32
  - **non-lidar features first**, **lidar last**
- `outputs`: `(N, H*4)` float32
- metadata includes at least:
  - `horizon` (H)
  - `n_beams` (lidar length)

Loading/splitting is handled by `LocalTrajDataset` (from `diffusion_local_traj.py`).

## Model architecture (deterministic baseline)

Implemented in `deterministic_local_traj.py`:

- **Lidar encoder**: small 1D CNN (`LidarCNN1D`) → lidar embedding (default **64-d**)
- **Condition encoder**: MLP over `(non_lidar || lidar_emb)` → global condition vector
- **Temporal decoder**: small **Transformer** operating on a sequence of length `H`
  - global condition is broadcast to `(B, H, D)`
  - learned positional embeddings over time steps
  - per-step head outputs `(dx, dy, sin(dθ), cos(dθ))`

Output shape is always: **`(B, H, 4)`**

## Loss

The default training objective is:

\[
L = L_{\text{traj}} + \lambda_{\text{smooth}} L_{\text{smooth}}
\]

- **Trajectory loss** \(L_{\text{traj}}\): MSE over all 4 channels on `(B, H, 4)`
- **Smoothness loss** \(L_{\text{smooth}}\): second-difference penalty over time on **all 4 channels**
  - \(x[k+1] - 2x[k] + x[k-1]\)
- Default: `lambda_smooth = 0.01`

## Metrics

Validation reports:

- **loss** (total)
- **traj_mse** (raw MSE term)
- **smooth** (raw smoothness term)
- **ADE / FDE** computed on `(dx, dy)` using `traj_metrics` (same as diffusion code, but **single deterministic prediction**, not best-of-N).

## Training

Run training (example matches your current dataset names):

```bash
python3 train_deterministic_local_traj.py \
  --train-npz train_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz \
  --val-npz   val_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz \
  --save-dir  models_deterministic_local_traj \
  --batch-size 256 \
  --epochs 30 \
  --lambda-smooth 0.01
```

### Useful knobs

- **Smoothness strength**: `--lambda-smooth 0.01`
- **Transformer capacity**:
  - `--model-dim` (default 128)
  - `--num-layers` (default 2)
  - `--num-heads` (default 4)
- **Regularization**: `--dropout` (default 0.1)
- **Optimizer**:
  - `--lr` (default 1e-3)
  - `--weight-decay` (default 1e-4)

## Outputs / artifacts

Saved under `--save-dir`:

- `best_model_deterministic_local_traj.pth` (chosen by best **val ADE**)
- `last_model_deterministic_local_traj.pth`
- Curves:
  - `training_curves_deterministic_local_traj_loss.png`
  - `training_curves_deterministic_local_traj_ade_fde.png`
- Qualitative plots (if `--plot-every > 0`):
  - `epochXXX_val_sampleYYYY_comparison.png`

The qualitative plot shows predicted vs expert trajectories in:

- local dx/dy and heading terms over time
- world-frame XY overlay, including current lidar points

## Files

- `deterministic_local_traj.py`: deterministic model + loss + evaluation utilities
- `train_deterministic_local_traj.py`: training loop + metrics + plotting + checkpointing

