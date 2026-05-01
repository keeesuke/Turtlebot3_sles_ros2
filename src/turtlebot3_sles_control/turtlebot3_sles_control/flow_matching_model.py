#!/usr/bin/env python3
"""
Flow Matching Policy model for local trajectory prediction.

Implements Optimal-Transport Conditional Flow Matching (OT-CFM):
  Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
  Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data
               with Rectified Flow", ICLR 2023

Inspired by the streaming variant in:
  Ancha et al., "Streaming Flow Policy"
  https://github.com/siddancha/streaming-flow-policy

Key idea (OT-CFM):
  Training:
    x_0 ~ N(0, I)                     (noise / source)
    x_1 = trajectory                  (data / target)
    t   ~ U[0, 1]
    x_t = (1 - t) * x_0 + t * x_1   (linear interpolant)
    v*  = x_1 - x_0                  (constant target velocity along the flow)
    loss = MSE(v_θ(x_t, t, cond), v*)

  Inference (ODE integration from t=0 → t=1):
    x_0 ~ N(0, I)
    x_{t+dt} = x_t + v_θ(x_t, t, cond) · dt   [Euler]
             or via RK4 for higher accuracy

Architecture:
  - Same observation encoder as diffusion_policy_model.py:
      LidarCNN1D(lidar) || MLP(non_lidar + lidar_emb) -> global_cond
  - Same ConditionalUnet1D backbone (now predicts velocity, not noise)
  - No external ODE solver needed — Euler/RK4 implemented in pure PyTorch

Dataset interface is identical to LocalTrajDataset in diffusion_policy_model.py.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse all building blocks from the diffusion policy module
from diffusion_policy_model import (
    ConditionalUnet1D,
    LidarCNN1D,
    LocalTrajDataset,
    StructuredConditionEncoder,
    traj_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FlowMatchingConfig:
    horizon: int
    non_lidar_dim: int
    n_beams: int
    history_len: int = 10                    # must match dataset history_len
    include_heading_diff: bool = False       # must match dataset flag
    action_dim: int = 4                      # (dx, dy, sin, cos)
    lidar_emb_dim: int = 64
    global_cond_dim: int = 256               # encoded state -> UNet global cond
    diffusion_step_embed_dim: int = 128      # reused for time embedding in UNet
    down_dims: Tuple = (256, 512, 1024)
    kernel_size: int = 3
    n_groups: int = 8
    cond_predict_scale: bool = False
    num_inference_steps: int = 100           # Euler steps at inference
    integrator: str = "euler"               # "euler" or "rk4"
    sigma_min: float = 1e-4                  # small noise floor at t=1 (improves training stability)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class FlowMatchingPolicyModel(nn.Module):
    """
    OT-CFM Policy for local trajectory prediction.

    Observation encoding (global conditioning):
      LidarCNN1D(lidar)              -> (B, lidar_emb_dim)
      MLP(non_lidar || lidar_emb)    -> (B, global_cond_dim)

    Velocity prediction:
      ConditionalUnet1D(x_t, t, global_cond) -> v_θ(x_t, t, cond)
        input_dim  = action_dim = 4
        UNet horizon = cfg.horizon

    Training objective:
      Given clean trajectory x_1 and noise x_0 ~ N(0,I):
        t  ~ U[0,1]
        x_t = (1-t)*x_0 + t*x_1
        v* = x_1 - x_0              (OT-CFM target velocity, constant along path)
        loss = MSE(v_θ(x_t, t, cond), v*)

    Inference:
      Euler:  x_{t+dt} = x_t + v_θ(x_t, t, cond) * dt
      RK4:    standard 4th-order Runge-Kutta
    """

    def __init__(self, cfg: FlowMatchingConfig):
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

        # Velocity network (UNet backbone)
        self.velocity_net = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=cfg.global_cond_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=list(cfg.down_dims),
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
            cond_predict_scale=cfg.cond_predict_scale,
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("FlowMatchingPolicyModel parameters: %e", n_params)

    def encode_condition(self, non_lidar: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """(B, non_lidar_dim), (B, n_beams) -> (B, global_cond_dim)"""
        return self.cond_enc(non_lidar, lidar)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, float],
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_t:        (B, H, action_dim)  interpolated trajectory at time t
        t:          scalar or (B,) tensor in [0, 1]
        global_cond:(B, global_cond_dim)
        returns:    (B, H, action_dim)  predicted velocity
        """
        B = x_t.shape[0]
        device = x_t.device

        if not torch.is_tensor(t):
            t_tensor = torch.full((B,), float(t), dtype=torch.float32, device=device)
        elif torch.is_tensor(t) and t.ndim == 0:
            t_tensor = t.unsqueeze(0).expand(B).to(device=device, dtype=torch.float32)
        else:
            t_tensor = t.to(device=device, dtype=torch.float32)
            if t_tensor.shape[0] == 1:
                t_tensor = t_tensor.expand(B)

        # ConditionalUnet1D uses its diffusion_step_encoder for the time embedding.
        # We pass t scaled to [0, num_train_timesteps) for the sinusoidal embedding —
        # here we keep t in [0,1] and the encoder handles it as a continuous scalar.
        return self.velocity_net(x_t, t_tensor, global_cond=global_cond)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        non_lidar: torch.Tensor,
        lidar: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        OT-CFM training loss.

        non_lidar:   (B, non_lidar_dim)
        lidar:       (B, n_beams)
        target_flat: (B, H*4)  clean trajectory x_1
        returns dict with 'loss' key (scalar)
        """
        B = non_lidar.shape[0]
        H = self.cfg.horizon
        device = non_lidar.device

        x_1 = target_flat.view(B, H, self.cfg.action_dim)   # (B, H, 4) clean traj

        # Source: pure Gaussian noise
        x_0 = torch.randn_like(x_1)

        # Random time in (0, 1) — avoid exact 0 and 1 for stability
        t = torch.rand(B, device=device)                     # (B,)

        # Linear interpolant: x_t = (1-t)*x_0 + t*x_1
        t_bcast = t.view(B, 1, 1)
        x_t = (1.0 - t_bcast) * x_0 + t_bcast * x_1

        # Optional sigma_min noise floor: x_t += sigma_min * eps
        # Keeps training numerically stable near t=1 without changing the target.
        if self.cfg.sigma_min > 0:
            x_t = x_t + self.cfg.sigma_min * torch.randn_like(x_t)

        # OT-CFM target velocity (constant along the linear path)
        v_target = x_1 - x_0                                # (B, H, 4)

        # Encode conditioning
        global_cond = self.encode_condition(non_lidar, lidar)

        # Predict velocity
        v_pred = self.predict_velocity(x_t, t, global_cond)  # (B, H, 4)

        loss = F.mse_loss(v_pred, v_target)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Inference (ODE integration: t=0 → t=1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_trajectory(
        self,
        non_lidar: torch.Tensor,
        lidar: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Integrate the learned ODE from t=0 (noise) to t=1 (trajectory).

        non_lidar: (B, non_lidar_dim)
        lidar:     (B, n_beams)
        returns:   (B, H, action_dim)  predicted clean trajectory
        """
        B = non_lidar.shape[0]
        H = self.cfg.horizon
        device = non_lidar.device
        steps = num_inference_steps or self.cfg.num_inference_steps

        global_cond = self.encode_condition(non_lidar, lidar)  # (B, cond_dim)

        # Start from pure noise
        x = torch.randn(
            (B, H, self.cfg.action_dim), device=device, generator=generator
        )

        dt = 1.0 / steps
        integrator = self.cfg.integrator

        for i in range(steps):
            t = i * dt  # current time in [0, 1)

            if integrator == "euler":
                v = self.predict_velocity(x, t, global_cond)
                x = x + v * dt

            elif integrator == "rk4":
                # 4th-order Runge-Kutta
                k1 = self.predict_velocity(x,               t,          global_cond)
                k2 = self.predict_velocity(x + 0.5*dt*k1,  t + 0.5*dt, global_cond)
                k3 = self.predict_velocity(x + 0.5*dt*k2,  t + 0.5*dt, global_cond)
                k4 = self.predict_velocity(x + dt*k3,       t + dt,     global_cond)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            else:
                raise ValueError(f"Unknown integrator: {integrator!r}. Use 'euler' or 'rk4'.")

        return x  # (B, H, 4) at t=1


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_epoch(
    model: FlowMatchingPolicyModel,
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

        # Heading error on the best sample
        ade_per = torch.norm(
            samples.view(B, num_samples, horizon, action_dim)[..., :2]
            - target.view(B, 1, horizon, action_dim)[..., :2],
            dim=-1,
        ).mean(dim=-1)   # (B, num_samples)
        best_idx = ade_per.argmin(dim=1)
        best_samples = samples[torch.arange(B, device=device), best_idx]
        hm = traj_metrics(best_samples, target, horizon=horizon, best_of_n=False, action_dim=action_dim)
        accum["heading_err"] += float(hm["heading_err"])

        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in accum.items()}


__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingPolicyModel",
    "evaluate_epoch",
    "LocalTrajDataset",
    "traj_metrics",
]
