#!/usr/bin/env python3
"""
Diffusion Policy model for local trajectory prediction.

Architecture follows the DDPM-based Diffusion Policy from:
  Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  https://github.com/real-stanford/diffusion_policy

Key components (self-contained, no einops dependency):
  - SinusoidalPosEmb       : timestep embedding
  - Conv1dBlock            : Conv1d -> GroupNorm -> Mish
  - Downsample1d / Upsample1d
  - ConditionalResidualBlock1D : FiLM-conditioned residual block
  - ConditionalUnet1D      : U-Net noise predictor operating on (B, T, action_dim)
  - LidarCNN1D             : 1D CNN encoder for lidar scans
  - DiffusionPolicyModel   : full model (encoder + UNet) with train/inference helpers

Dataset interface is identical to LocalTrajDataset in deterministic_local_traj.py:
  Each sample: (non_lidar [F], lidar [n_beams]) -> trajectory (horizon * 4)
  Trajectory channels: (dx, dy, sin(dtheta), cos(dtheta))
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset (identical interface to deterministic_local_traj.LocalTrajDataset)
# ---------------------------------------------------------------------------

class LocalTrajDataset(Dataset):
    """
    Loads flattened inputs/outputs from .npz, splits lidar from non-lidar.
    Identical schema to the dataset used by the deterministic baseline.
    """

    def __init__(self, npz_file: str):
        data = np.load(npz_file, allow_pickle=True)
        if "inputs" not in data.files or "outputs" not in data.files:
            raise KeyError("Expected keys 'inputs' and 'outputs' in dataset npz")

        self.inputs = data["inputs"].astype(np.float32)
        self.outputs = data["outputs"].astype(np.float32)

        # action_dim: 4 for (dx,dy,sin,cos) trajectory, 6 for (dx,dy,sin,cos,v,w) waypoints
        self.action_dim = int(data["action_dim"]) if "action_dim" in data.files else 4
        self.horizon = int(data["horizon"]) if "horizon" in data.files else (self.outputs.shape[1] // self.action_dim)
        self.target_dim = self.outputs.shape[1]

        self.n_beams = int(data["n_beams"]) if "n_beams" in data.files else None
        if self.n_beams is None:
            raise KeyError("Dataset missing 'n_beams' (needed to split lidar from inputs).")

        if self.inputs.ndim != 2:
            raise ValueError("inputs must be 2D (N, input_dim)")
        if self.outputs.ndim != 2:
            raise ValueError("outputs must be 2D (N, horizon*action_dim)")
        if self.outputs.shape[1] != self.horizon * self.action_dim:
            raise ValueError(
                f"outputs dim mismatch: got {self.outputs.shape[1]}, "
                f"expected horizon*action_dim = {self.horizon}*{self.action_dim} = {self.horizon * self.action_dim}"
            )
        if self.inputs.shape[1] < self.n_beams:
            raise ValueError("input_dim < n_beams; cannot split lidar")

        self.non_lidar_dim = self.inputs.shape[1] - self.n_beams
        if self.non_lidar_dim <= 0:
            raise ValueError("non_lidar_dim <= 0; check dataset schema")

        # Validate sin^2 + cos^2 = 1 for heading channels (indices 2,3 in both action_dim=4 and 6)
        outs = self.outputs.reshape(self.outputs.shape[0], self.horizon, self.action_dim)
        unit = outs[..., 2] ** 2 + outs[..., 3] ** 2
        max_err = float(np.max(np.abs(unit - 1.0)))
        if not np.isfinite(max_err) or max_err > 1e-3:
            raise ValueError("Dataset outputs failed sin/cos unit check; regenerate dataset.")

        self.meta = {k: data[k] for k in data.files if k not in ("inputs", "outputs")}

        print(f"Loaded dataset {npz_file}")
        print(f"  inputs:  {self.inputs.shape} (non_lidar_dim={self.non_lidar_dim}, n_beams={self.n_beams})")
        print(f"  outputs: {self.outputs.shape} (horizon={self.horizon}, action_dim={self.action_dim})")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]
        y = self.outputs[idx]
        non_lidar = torch.from_numpy(x[: self.non_lidar_dim]).float()
        lidar = torch.from_numpy(x[self.non_lidar_dim :]).float()
        target = torch.from_numpy(y).float()
        return non_lidar, lidar, target


# ---------------------------------------------------------------------------
# Trajectory metrics (same as deterministic baseline)
# ---------------------------------------------------------------------------

def traj_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    horizon: int,
    best_of_n: bool = False,
    action_dim: int = 4,
) -> Dict[str, float]:
    """
    Compute trajectory metrics for offline evaluation of a generative model.

    pred:
      - best_of_n=False: (B, horizon*action_dim)
      - best_of_n=True : (B, N, horizon*action_dim)
    gt: (B, horizon*action_dim)

    Channels 0,1 are always (dx, dy); channels 2,3 are always (sin_dθ, cos_dθ).
    action_dim=4: (dx, dy, sin, cos)          — sim / trajectory format
    action_dim=6: (dx, dy, sin, cos, v, w)    — real-world waypoint format
    """
    with torch.no_grad():
        if not best_of_n:
            p = pred.view(pred.shape[0], horizon, action_dim)
            g = gt.view(gt.shape[0], horizon, action_dim)
            # XY error
            d = torch.norm(p[..., :2] - g[..., :2], dim=-1)   # (B, H)
            ade = torch.mean(d).item()
            fde = torch.mean(d[:, -1]).item()
            # Heading error: arccos(sin_p*sin_g + cos_p*cos_g)
            cos_sim = (p[..., 2] * g[..., 2] + p[..., 3] * g[..., 3]).clamp(-1.0, 1.0)
            heading_err = torch.acos(cos_sim).mean().item()
            return {"ade": ade, "fde": fde, "heading_err": heading_err}

        b, n, td = pred.shape
        p = pred.view(b, n, horizon, action_dim)
        g = gt.view(b, 1, horizon, action_dim)

        # Per-sample XY distance to GT: (B, N, H)
        d = torch.norm(p[..., :2] - g[..., :2], dim=-1)

        # mean ADE/FDE (average over all samples — is the distribution centred right?)
        ade = torch.mean(d).item()
        fde = torch.mean(d[:, :, -1]).item()

        # minADE / minFDE — best sample per scene (primary generative metric)
        ade_per_sample = d.mean(dim=-1)            # (B, N)
        best_idx = ade_per_sample.argmin(dim=1)    # (B,)
        idx_expand = best_idx.view(b, 1, 1).expand(b, 1, horizon)
        d_best = d.gather(1, idx_expand).squeeze(1)  # (B, H)
        min_ade = d_best.mean().item()
        min_fde = d_best[:, -1].mean().item()

        # Diversity: mean pairwise L2 between samples (averaged over batch)
        # p_xy: (B, N, H*2) for pairwise computation
        p_xy = p[..., :2].reshape(b, n, horizon * 2)
        # (B, N, N) pairwise distances
        diff = p_xy.unsqueeze(2) - p_xy.unsqueeze(1)           # (B, N, N, H*2)
        pair_dist = torch.norm(diff, dim=-1)                   # (B, N, N)
        # upper triangle only (avoid double-counting and diagonal)
        mask = torch.triu(torch.ones(n, n, device=pred.device, dtype=torch.bool), diagonal=1)
        diversity = pair_dist[:, mask].mean().item()

        return {
            "ade": ade,
            "fde": fde,
            "min_ade": min_ade,
            "min_fde": min_fde,
            "diversity": diversity,
        }


# ---------------------------------------------------------------------------
# U-Net building blocks (from real-stanford/diffusion_policy, no einops)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish"""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual block with FiLM conditioning (https://arxiv.org/abs/1709.07871).
    Predicts per-channel scale and bias (or bias only) from cond vector.

    x:    (B, in_channels, T)
    cond: (B, cond_dim)
    out:  (B, out_channels, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)           # (B, cond_channels)
        embed = embed.unsqueeze(-1)               # (B, cond_channels, 1)

        if self.cond_predict_scale:
            scale = embed[:, : self.out_channels, :]
            bias = embed[:, self.out_channels :, :]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    Conditional 1D U-Net noise predictor.

    Adapted from real-stanford/diffusion_policy (Planning with Diffusion UNet).
    Operates on trajectory tensors of shape (B, T, input_dim).

    Conditioning:
      - Diffusion timestep: sinusoidal embedding injected at every residual block.
      - global_cond: optional (B, global_cond_dim) tensor concatenated with timestep emb.
      - local_cond:  optional (B, T, local_cond_dim) per-timestep conditioning.
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: List[int] = None,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Optional local conditioning encoder (two blocks: one for down, one for up)
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    local_cond_dim, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    local_cond_dim, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("ConditionalUnet1D parameters: %e", n_params)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sample:      (B, T, input_dim)  - noisy trajectory
        timestep:    (B,) or scalar     - diffusion step
        local_cond:  (B, T, local_cond_dim) or None
        global_cond: (B, global_cond_dim) or None
        returns:     (B, T, input_dim)  - predicted noise (or x0)
        """
        # (B, T, C) -> (B, C, T) for Conv1d
        x = sample.permute(0, 2, 1).contiguous()

        # Timestep embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)  # (B, dsed)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Local conditioning
        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            lc = local_cond.permute(0, 2, 1).contiguous()  # (B, C_lc, T)
            resnet, resnet2 = self.local_cond_encoder
            h_local.append(resnet(lc, global_feature))
            h_local.append(resnet2(lc, global_feature))

        # Encoder
        skips = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        # Decoder
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            skip = skips.pop()
            if x.shape[-1] > skip.shape[-1]:
                x = x[..., :skip.shape[-1]]
            x = torch.cat((x, skip), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        # (B, C, T) -> (B, T, C)
        return x.permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
# Lidar encoder
# ---------------------------------------------------------------------------

class LidarCNN1D(nn.Module):
    """1D CNN encoder: (B, n_beams) -> (B, emb_dim)"""

    def __init__(self, n_beams: int, emb_dim: int = 64):
        super().__init__()
        self.n_beams = n_beams
        self.emb_dim = emb_dim

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64, emb_dim), nn.ReLU())

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(lidar.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Structured condition encoder
# ---------------------------------------------------------------------------

class TemporalHistoryEncoder(nn.Module):
    """
    1D CNN over a short temporal sequence.

    Input:  (B, T, C)   - T timesteps, C features each
    Output: (B, emb_dim)

    Uses Conv1d over the time axis so temporal ordering is exploited.
    """

    def __init__(self, in_channels: int, history_len: int, emb_dim: int):
        super().__init__()
        # Conv1d expects (B, C, T); we'll permute inside forward.
        # Two conv layers then adaptive pool to a fixed-size embedding.
        mid = max(emb_dim, in_channels * 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(mid, emb_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1).contiguous()
        return self.flatten(self.net(x))   # (B, emb_dim)


class StructuredConditionEncoder(nn.Module):
    """
    Encodes the flat non_lidar vector by slicing it into semantically distinct
    groups and applying a dedicated encoder to each — without any dataset changes.

    Input layout (matches create_dataset_imitation_local_traj.py):
      [0 : 6]                          current state  (x_t, y_t, sinθ_t, cosθ_t, v_t, ω_t)
      [6 : 6+H*6]                      state history  (H × 6), oldest→newest, world frame
      [6+H*6 : 6+H*6+H*2]             cmd history    (H × 2), oldest→newest
      [6+H*6+H*2 : 6+H*6+H*2+2]       goal           (x_rel, y_rel) already in robot frame
      [... : ...+2]  (optional)        heading diff   (sin, cos) if include_heading_diff

    Frame normalisation (applied before encoding, no dataset changes):
      State history (x_k, y_k, θ_k) are stored in world frame in the dataset.
      We transform them on-the-fly into the current robot frame at time t:
        dx_l =  (x_k - x_t)*cosθ_t + (y_k - y_t)*sinθ_t
        dy_l = -(x_k - x_t)*sinθ_t + (y_k - y_t)*cosθ_t
        sinΔθ = sin(θ_k - θ_t),  cosΔθ = cos(θ_k - θ_t)
      Current state: drop (x_t, y_t) — absolute world position is not useful.
      Encode: (sinθ_t, cosθ_t, v_t, ω_t) for the current step.

    Dedicated encoders (all inputs in local/robot frame):
      current state  → Linear(4, 32)                           (sinθ, cosθ, v, ω)
      state history  → TemporalHistoryEncoder(6, H, 64)        (dx, dy, sinΔθ, cosΔθ, v, ω)
      cmd history    → TemporalHistoryEncoder(2, H, 32)        (v_cmd, w_cmd)
      goal           → Linear(2, 16)                           already local frame
      heading diff   → Linear(2, 8)  (only if include_heading_diff)
      lidar          → LidarCNN1D(n_beams, lidar_emb_dim)      already local frame

    All embeddings are concatenated then projected to global_cond_dim.
    """

    # Fixed per-step feature dims (must match dataset creation)
    _STATE_DIM = 6   # (x, y, sinθ, cosθ, v, ω) as stored in dataset
    _CMD_DIM   = 2   # (v_cmd, w_cmd)

    def __init__(
        self,
        n_beams: int,
        history_len: int,
        include_heading_diff: bool,
        global_cond_dim: int,
        lidar_emb_dim: int = 64,
    ):
        super().__init__()
        self.history_len          = history_len
        self.include_heading_diff = include_heading_diff

        # Slice offsets into the local-frame non_lidar vector
        H = history_len
        self._off_curr_end  = 2
        self._off_shist_end = 2 + H * self._STATE_DIM
        self._off_chist_end = 2 + H * self._STATE_DIM + H * self._CMD_DIM
        self._off_goal_end  = self._off_chist_end + 2
        self._off_head_end  = self._off_goal_end + (2 if include_heading_diff else 0)

        # Per-group encoders — all in local/robot frame
        # Current state: (v_t, ω_t) — 2 dims; sinθ_t=0, cosθ_t=1 in canonical frame, carry no info
        self.curr_enc  = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        # State history: (dx, dy, sinΔθ, cosΔθ, v, ω) per step — 6 dims, local frame
        self.shist_enc = TemporalHistoryEncoder(self._STATE_DIM, H, emb_dim=64)
        # Cmd history: (v_cmd, w_cmd) — already body-frame, no change
        self.chist_enc = TemporalHistoryEncoder(self._CMD_DIM, H, emb_dim=32)
        # Goal: (x_rel, y_rel) — already in robot frame from dataset
        self.goal_enc  = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.head_enc  = nn.Sequential(nn.Linear(2, 8), nn.ReLU()) if include_heading_diff else None
        self.lidar_enc = LidarCNN1D(n_beams=n_beams, emb_dim=lidar_emb_dim)

        # Projection to global_cond_dim
        concat_dim = 32 + 64 + 32 + 16 + (8 if include_heading_diff else 0) + lidar_emb_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, global_cond_dim),
            nn.ReLU(),
            nn.Linear(global_cond_dim, global_cond_dim),
            nn.ReLU(),
        )

    def forward(self, non_lidar: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """
        non_lidar: (B, non_lidar_dim)  — all fields pre-transformed to local/robot frame
        lidar:     (B, n_beams)
        returns:   (B, global_cond_dim)

        Expected layout (set by dataset creators):
          curr  [0:2]          = (v_t/v_max, w_t/w_max)
          shist [2:2+H*6]     = H × (dx_l, dy_l, sin_dθ, cos_dθ, v/v_max, w/w_max)
          chist [..:..+H*2]   = H × (v_cmd/v_max, w_cmd/w_max)
          goal  [..:..+2]     = (g_x_rel, g_y_rel) in robot frame
        """
        H = self.history_len

        curr  = non_lidar[:, :self._off_curr_end]                    # (B, 2): (v_t, w_t)
        shist = non_lidar[:, self._off_curr_end:self._off_shist_end] # (B, H*6)
        chist = non_lidar[:, self._off_shist_end:self._off_chist_end]# (B, H*2)
        goal  = non_lidar[:, self._off_chist_end:self._off_goal_end] # (B, 2)

        shist = shist.view(-1, H, self._STATE_DIM)   # (B, H, 6): (dx,dy,sin_dθ,cos_dθ,v,ω)
        chist = chist.view(-1, H, self._CMD_DIM)     # (B, H, 2)

        curr_local = curr   # (B, 2): (v_t, w_t) — pose is identity

        # History already in local frame — no transform needed
        parts = [
            self.curr_enc(curr_local),   # (B, 32)
            self.shist_enc(shist),       # (B, 64)
            self.chist_enc(chist),       # (B, 32)
            self.goal_enc(goal),         # (B, 16)
            self.lidar_enc(lidar),       # (B, lidar_emb_dim)
        ]

        if self.include_heading_diff and self.head_enc is not None:
            head = non_lidar[:, self._off_goal_end:self._off_head_end]   # (B, 2)
            parts.append(self.head_enc(head))                            # (B, 8)

        return self.proj(torch.cat(parts, dim=1))


# ---------------------------------------------------------------------------
# Config & top-level model
# ---------------------------------------------------------------------------

@dataclass
class DiffusionPolicyConfig:
    horizon: int
    non_lidar_dim: int
    n_beams: int
    history_len: int = 10                  # must match dataset history_len
    include_heading_diff: bool = False     # must match dataset flag
    action_dim: int = 4                    # (dx, dy, sin, cos)
    lidar_emb_dim: int = 64
    global_cond_dim: int = 256             # encoded state -> UNet global cond
    diffusion_step_embed_dim: int = 128
    down_dims: Tuple = (256, 512, 1024)
    kernel_size: int = 3
    n_groups: int = 8
    cond_predict_scale: bool = False
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"       # "epsilon" or "sample"


class DiffusionPolicyModel(nn.Module):
    """
    Diffusion Policy for local trajectory prediction.

    Observation encoding (global conditioning):
      - LidarCNN1D(lidar)              -> (B, lidar_emb_dim)
      - MLP(non_lidar || lidar_emb)    -> (B, global_cond_dim)

    Noise prediction:
      - ConditionalUnet1D(noisy_traj, t, global_cond) -> predicted noise / x0
        input_dim = action_dim = 4
        UNet horizon = cfg.horizon

    Forward (training):
      Returns predicted noise/x0 for given noisy trajectory + timestep.

    Inference:
      sample_trajectory() runs the full DDPM reverse process.
    """

    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg

        # Structured condition encoder: dedicated per-group encoders
        self.cond_enc = StructuredConditionEncoder(
            n_beams=cfg.n_beams,
            history_len=cfg.history_len,
            include_heading_diff=cfg.include_heading_diff,
            global_cond_dim=cfg.global_cond_dim,
            lidar_emb_dim=cfg.lidar_emb_dim,
        )

        # U-Net noise predictor
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=cfg.global_cond_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=list(cfg.down_dims),
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
            cond_predict_scale=cfg.cond_predict_scale,
        )

        # DDPM noise scheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            prediction_type=cfg.prediction_type,
        )

    def encode_condition(self, non_lidar: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """(B, non_lidar_dim), (B, n_beams) -> (B, global_cond_dim)"""
        return self.cond_enc(non_lidar, lidar)

    def forward(
        self,
        noisy_traj: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy_traj:  (B, H, action_dim)
        timesteps:   (B,)
        global_cond: (B, global_cond_dim)
        returns:     (B, H, action_dim)  predicted noise or x0
        """
        return self.unet(noisy_traj, timesteps, global_cond=global_cond)

    def compute_loss(
        self,
        non_lidar: torch.Tensor,
        lidar: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard DDPM training loss.

        non_lidar:   (B, non_lidar_dim)
        lidar:       (B, n_beams)
        target_flat: (B, H*4)  clean trajectory
        returns dict with 'loss' key (scalar)
        """
        b = non_lidar.shape[0]
        h = self.cfg.horizon
        device = non_lidar.device

        trajectory = target_flat.view(b, h, self.cfg.action_dim)  # (B, H, 4)
        global_cond = self.encode_condition(non_lidar, lidar)       # (B, cond_dim)

        # Sample noise and random timesteps
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (b,), device=device, dtype=torch.long,
        )

        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        noisy_traj = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Predict noise (or x0)
        pred = self.forward(noisy_traj, timesteps, global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        loss = F.mse_loss(pred, target)
        return {"loss": loss}

    @torch.no_grad()
    def sample_trajectory(
        self,
        non_lidar: torch.Tensor,
        lidar: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Run the full DDPM reverse denoising process.

        non_lidar: (B, non_lidar_dim)
        lidar:     (B, n_beams)
        returns:   (B, H, action_dim)  denoised trajectory
        """
        b = non_lidar.shape[0]
        h = self.cfg.horizon
        device = non_lidar.device
        steps = num_inference_steps or self.cfg.num_inference_steps

        global_cond = self.encode_condition(non_lidar, lidar)

        # Start from pure noise
        traj = torch.randn(
            (b, h, self.cfg.action_dim), device=device, generator=generator
        )

        self.noise_scheduler.set_timesteps(steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.forward(traj, t, global_cond)
            traj = self.noise_scheduler.step(
                model_output, t, traj, generator=generator
            ).prev_sample

        return traj  # (B, H, 4)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_epoch(
    model: DiffusionPolicyModel,
    loader: DataLoader,
    device: str,
    horizon: int,
    num_inference_steps: int = 10,
    num_samples: int = 8,
    max_batches: int = 0,
    action_dim: int = 4,
) -> Dict[str, float]:
    """
    Evaluate on the validation set.

    For each batch, draws `num_samples` independent trajectory samples and computes:
      loss        — MSE training proxy (fast, single forward pass)
      ade / fde   — mean ADE/FDE over all samples (is distribution centred?)
      min_ade     — best-of-N ADE  (primary metric for generative models)
      min_fde     — best-of-N FDE
      diversity   — mean pairwise L2 between samples (mode-collapse detector)
      heading_err — mean angular error of the best sample (radians)

    max_batches: cap number of val batches (0 = use all).
    """
    model.eval()
    accum: Dict[str, float] = {
        "loss": 0.0, "ade": 0.0, "fde": 0.0,
        "min_ade": 0.0, "min_fde": 0.0,
        "diversity": 0.0, "heading_err": 0.0,
    }
    n_batches = 0

    for non_lidar, lidar, target in loader:
        if max_batches > 0 and n_batches >= max_batches:
            break
        non_lidar = non_lidar.to(device)
        lidar     = lidar.to(device)
        target    = target.to(device)
        B         = non_lidar.shape[0]

        # Fast proxy loss (no full denoising needed)
        accum["loss"] += float(model.compute_loss(non_lidar, lidar, target)["loss"].item())

        # Draw num_samples independent trajectories: (B, num_samples, H*4)
        samples = torch.stack(
            [model.sample_trajectory(non_lidar, lidar, num_inference_steps=num_inference_steps)
             .reshape(B, -1)
             for _ in range(num_samples)],
            dim=1,
        )

        m = traj_metrics(samples, target, horizon=horizon, best_of_n=True, action_dim=action_dim)
        for k in ("ade", "fde", "min_ade", "min_fde", "diversity"):
            accum[k] += float(m[k])

        # Heading error on the best sample (best_idx by min_ade)
        ade_per = torch.norm(
            samples.view(B, num_samples, horizon, action_dim)[..., :2]
            - target.view(B, 1, horizon, action_dim)[..., :2],
            dim=-1,
        ).mean(dim=-1)   # (B, num_samples)
        best_idx = ade_per.argmin(dim=1)  # (B,)
        best_samples = samples[torch.arange(B, device=device), best_idx]
        hm = traj_metrics(best_samples, target, horizon=horizon, best_of_n=False, action_dim=action_dim)
        accum["heading_err"] += float(hm["heading_err"])

        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in accum.items()}


__all__ = [
    "LocalTrajDataset",
    "traj_metrics",
    "DiffusionPolicyConfig",
    "DiffusionPolicyModel",
    "evaluate_epoch",
    "LidarCNN1D",
    "ConditionalUnet1D",
    "ConditionalResidualBlock1D",
    "SinusoidalPosEmb",
]
