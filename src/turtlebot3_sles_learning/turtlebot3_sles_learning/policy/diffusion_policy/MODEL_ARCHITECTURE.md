# Diffusion / Flow Matching Policy — Architecture Reference

## Overview

Two generative model variants for **local trajectory prediction** on a mobile robot:

| Model | File | Training objective | Inference |
|---|---|---|---|
| Diffusion Policy (DDPM) | `diffusion_policy_model.py` | Predict noise ε | 100 reverse denoising steps |
| Flow Matching (OT-CFM) | `flow_matching_model.py` | Predict velocity field v | 100 Euler/RK4 integration steps |

Both share the same **observation encoder** and **ConditionalUnet1D backbone**.  
Dataset creation: `create_dataset_imitation_local_traj.py`  
Training scripts: `train_diffusion_policy.py`, `train_flow_matching.py`

---

## Dataset Input

Each sample is anchored at control timestep `t` (~10 Hz, `dt = 0.1 s`).  
The raw dataset stores a flat `(448,)` input vector and a `(80,)` output vector.

### Raw input vector layout `(448,)`

```
Offset    Dim   Feature                           Frame          Normalisation
────────────────────────────────────────────────────────────────────────────────
[0:2]       2   x_t, y_t  (current position)      World          raw metres
[2:4]       2   sinθ_t, cosθ_t  (current heading) World          [-1, 1]
[4:6]       2   v_t, ω_t  (current velocity)      Body           v/v_max, ω/w_max

[6:66]     60   State history × 10 steps           World          same as above
                oldest → newest, each step:
                (x_k, y_k, sinθ_k, cosθ_k, v_k, ω_k)

[66:86]    20   Command history × 10 steps         Body           v/v_max, ω/w_max
                oldest → newest, each step:
                (v_cmd_k, w_cmd_k)

[86:88]     2   Goal position (x_rel, y_rel)       Current robot  raw metres (relative)
                                                   frame (local)

[88:448]  360   Lidar scan (360 beams, 1° res.)    Current robot  pre-normalised [0, 1]
                                                   frame (local)  invalid beam → 1.0
────────────────────────────────────────────────────────────────────────────────
Total:    448
```

**Notes:**
- `v_max = 0.26 m/s`, `w_max = 1.5 rad/s`
- State history and current `(x_t, y_t)` are in **world frame** as stored — the encoder transforms these to local frame at runtime (no dataset changes needed)
- Lidar and goal are already in the current robot frame as stored

---

## Observation Encoder — `StructuredConditionEncoder`

The flat input is sliced into semantic groups. **All world-frame quantities are converted to the current robot frame on-the-fly** before encoding.

### Frame transform (applied at runtime)

State history positions `(x_k, y_k)` in world frame → local frame:
```
dx_l =  (x_k − x_t)·cosθ_t + (y_k − y_t)·sinθ_t
dy_l = −(x_k − x_t)·sinθ_t + (y_k − y_t)·cosθ_t
```

State history heading `θ_k` absolute → relative:
```
sinΔθ_k = sinθ_k·cosθ_t − cosθ_k·sinθ_t
cosΔθ_k = cosθ_k·cosθ_t + sinθ_k·sinθ_t
```

Current `(x_t, y_t)` world position: **dropped** — absolute map position carries no useful signal for a local policy.

### Per-group encoders

All inputs below are in the current robot frame after transform.

```
Group              Local-frame input                        Encoder                         Out dim
────────────────────────────────────────────────────────────────────────────────────────────────────
Current state      (v_t, ω_t)                               Linear(2 → 32) + ReLU            32
                   [world xy dropped; sinθ_t=0, cosθ_t=1
                    in canonical frame — carry no info]

State history      (dx_k, dy_k, sinΔθ_k, cosΔθ_k,          TemporalHistoryEncoder            64
                    v_k, ω_k) × 10 steps                    Conv1d(6→mid→64) over time
                   [transformed from world frame]            + AdaptiveAvgPool1d + Flatten

Command history    (v_cmd_k, w_cmd_k) × 10 steps            TemporalHistoryEncoder            32
                   [body frame, no transform needed]         Conv1d(2→mid→32) over time
                                                             + AdaptiveAvgPool1d + Flatten

Goal               (x_goal_rel, y_goal_rel)                 Linear(2 → 16) + ReLU            16
                   [already local frame in dataset]

Lidar scan         360 range values ∈ [0, 1]                LidarCNN1D                        64
                   [already local frame in dataset]         Conv1d(1→16, k=9, s=2)
                                                            Conv1d(16→32, k=7, s=2)
                                                            Conv1d(32→64, k=5, s=2)
                                                            AdaptiveAvgPool1d(1)
                                                            Linear(64 → 64) + ReLU
────────────────────────────────────────────────────────────────────────────────────────────────────
Concatenated                                                                                   208

Projection MLP     Linear(208 → 256) + ReLU
                   Linear(256 → 256) + ReLU                                               →  256  = global_cond
```

`TemporalHistoryEncoder` uses `Conv1d` over the **time axis** (T=10), so the model sees the history as an ordered temporal signal rather than a flat vector. This is strictly better than flat concatenation for a 10-step history.

---

## Backbone — `ConditionalUnet1D`

Operates on the trajectory tensor `(B, H=20, 4)`. Conditioned on `global_cond` at **every residual block** via FiLM modulation.

### Conditioning signal

```
t_embed  = SinusoidalPosEmb(t) → Linear → Mish → Linear   shape: (B, 128)
           [diffusion step index or flow matching time ∈ [0,1]]

global_feature = concat(t_embed, global_cond)              shape: (B, 128+256) = (B, 384)
```

### ConditionalResidualBlock1D (CResBlock)

Applied at every level of the UNet:
```
out = Conv1dBlock(x)                                   [Conv1d → GroupNorm → Mish]
out = out + Linear(global_feature).unsqueeze(-1)       [FiLM bias modulation]
out = Conv1dBlock(out)
out = out + residual_conv(x)                           [skip connection]
```

Optional `cond_predict_scale=True` predicts both scale and bias (full FiLM).

### UNet structure (default: `down_dims=[256, 512, 1024]`)

```
Input:  (B, H=20, 4)  →  permute  →  (B, 4, H=20)

Encoder (down path):
  Level 0:  CResBlock(4 → 256)  +  CResBlock(256 → 256)  →  skip₀  →  Downsample ÷2
  Level 1:  CResBlock(256 → 512)  +  CResBlock(512 → 512)  →  skip₁  →  Downsample ÷2
  Level 2:  CResBlock(512 → 1024)  +  CResBlock(1024 → 1024)  →  skip₂  (no downsample)

Bottleneck:
  CResBlock(1024 → 1024)  ×  2

Decoder (up path):
  Level 0:  cat(x, skip₂) → CResBlock(2048 → 512)  +  CResBlock(512 → 512)  →  Upsample ×2
  Level 1:  cat(x, skip₁) → CResBlock(1024 → 256)  +  CResBlock(256 → 256)  →  Upsample ×2

Final conv:
  Conv1dBlock(256 → 256)  +  Conv1d(256 → 4)

Output:  (B, 4, H=20)  →  permute  →  (B, H=20, 4)
```

Downsample: stride-2 `Conv1d`.  Upsample: stride-2 `ConvTranspose1d`.

---

## Diffusion Policy (DDPM)

### Training

```
x_1 = clean trajectory  (B, H, 4)
ε   ~ N(0, I)
t   ~ Uniform{1, ..., T}   T = 100

x_t = sqrt(ᾱ_t) · x_1 + sqrt(1 − ᾱ_t) · ε     ← DDPM forward process

ε_pred = UNet(x_t, t, global_cond)
loss   = MSE(ε_pred, ε)
```

Beta schedule: `squaredcos_cap_v2` (cosine). Prediction type: `epsilon`.

### Inference (100 reverse steps)

```
x_T ~ N(0, I)
for t = T, T−1, ..., 1:
    ε_pred    = UNet(x_t, t, global_cond)
    x_{t−1}   = DDPMScheduler.step(x_t, ε_pred, t)
return x_0   →  (B, H=20, 4)
```

---

## Flow Matching (OT-CFM)

### Training

```
x_0 ~ N(0, I)            ← noise source
x_1 = clean trajectory   ← data target
t   ~ Uniform[0, 1]

x_t     = (1 − t) · x_0 + t · x_1   ← linear interpolant (optimal transport path)
v_target = x_1 − x_0                 ← constant velocity along the straight path

v_pred = UNet(x_t, t, global_cond)
loss   = MSE(v_pred, v_target)
```

### Inference

**Euler** (default, 100 steps):
```
x_0 ~ N(0, I)
dt  = 1 / 100
for i = 0, 1, ..., 99:
    t         = i · dt
    v         = UNet(x_t, t, global_cond)
    x_{t+dt}  = x_t + v · dt
return x_1   →  (B, H=20, 4)
```

**RK4** (optional, same loop with 4-stage Runge-Kutta step, use `--integrator rk4`).

Flow matching advantage: straight transport paths → fewer inference steps needed (10–20 steps often sufficient vs 100 for DDPM).

---

## Output Trajectory

Shape: `(B, H=20, 4)`

Each of the 20 future waypoints at time `t + k·dt` (`k = 1..20`, `dt = 0.1 s`):

```
Channel 0:  dx_k         x-displacement from current position, in CURRENT ROBOT FRAME   [metres]
Channel 1:  dy_k         y-displacement from current position, in CURRENT ROBOT FRAME   [metres]
Channel 2:  sinΔθ_k      sin of heading change relative to current heading θ_t
Channel 3:  cosΔθ_k      cos of heading change relative to current heading θ_t
```

The output is a sequence of **relative waypoints in the robot's local frame** — not control commands and not world-frame positions.

### Converting to world frame (for visualisation)

```python
dx_world = dx_l * cos(θ_t) - dy_l * sin(θ_t)
dy_world = dx_l * sin(θ_t) + dy_l * cos(θ_t)
x_k      = x_t + dx_world
y_k      = y_t + dy_world
θ_k      = θ_t + arctan2(sinΔθ_k, cosΔθ_k)
```

### Deployment (receding-horizon MPC)

```
At each control cycle (~10 Hz):
  1. Observe current state + lidar
  2. Run encoder → global_cond
  3. Run full denoising / ODE integration → trajectory (H=20 waypoints)
  4. Execute first 1–3 waypoints via tracking controller (e.g. pure pursuit)
  5. Discard remainder, replan next cycle
```

The model self-corrects on every cycle. No streaming re-use of the ODE trajectory is needed at these speeds (v_max = 0.26 m/s).

---

## Parameter Count (default config)

| Component | Parameters |
|---|---|
| StructuredConditionEncoder | ~0.4 M |
| ConditionalUnet1D | ~10.0 M |
| **Total** | **~10.4 M** |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Local frame for all inputs | Frame-invariant: generalises across map positions and environments |
| Drop world `(x_t, y_t)` | Absolute position is meaningless for a local policy |
| Conv1d temporal encoder for history | 10-step history has exploitable temporal ordering; flat concat wastes it |
| Waypoint output, not (v, ω) commands | MPC expert produces smooth trajectories; command-level signal is noisier |
| Batch receding-horizon, not streaming | Vehicle speed (0.26 m/s) gives ample replanning time; streaming adds complexity with no benefit |
| Lidar as 1D CNN on raw ranges | Simple and effective; 2D lidar scan is a 1D angular signal |
