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
    collision_loss_weight: float = 0.0       # weight for differentiable obstacle penalty (0 = disabled)
    robot_radius: float = 0.15              # metres — used to shrink the safe margin from the wall
    collision_penalty_type: str = "relu"    # "relu": fires only past wall; "exp": soft proximity penalty
    collision_sigma: float = 0.3            # metres — decay width for exponential penalty


# ---------------------------------------------------------------------------
# Differentiable collision penalty
# ---------------------------------------------------------------------------

def compute_collision_loss(
    waypoints_xy: torch.Tensor,  # (B, H, 2)  local-frame cumulative positions (metres)
    lidar: torch.Tensor,         # (B, n_beams)  raw metres, 0 = max range
    lidar_max_range: float,
    robot_radius: float = 0.15,
    penalty_type: str = "relu",
    sigma: float = 0.3,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Penalize predicted waypoints near or past the nearest wall.

    penalty_type="relu":
      Fires only when the waypoint crosses (d_wall - robot_radius).
      penalty = ReLU(wp_dist - margin)²

    penalty_type="exp":
      Soft proximity penalty that grows exponentially as the waypoint
      approaches the wall, even before crossing it.
      clearance = margin - wp_dist   (positive = safe gap)
      penalty   = exp(-clearance / sigma)
      At clearance=sigma the penalty is 1/e; at the wall boundary it is 1;
      inside the wall it grows rapidly.
    """
    B, H, _ = waypoints_xy.shape
    n_beams = lidar.shape[1]

    dx = waypoints_xy[..., 0]  # (B, H)
    dy = waypoints_xy[..., 1]  # (B, H)

    wp_dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # (B, H)

    # Angle in [0, 2π)
    angle = torch.atan2(dy, dx) % (2.0 * math.pi)   # (B, H)

    # Fractional beam index
    frac_idx = angle / (2.0 * math.pi) * n_beams     # (B, H) in [0, n_beams)

    # Neighbouring beam indices (wrap around)
    idx_lo = frac_idx.long() % n_beams               # (B, H)
    idx_hi = (idx_lo + 1) % n_beams                  # (B, H)
    alpha = frac_idx - frac_idx.detach().floor()     # (B, H) fractional weight

    # Gather lidar distances at the two surrounding beams
    lidar_lo = lidar.gather(1, idx_lo.view(B, -1)).view(B, H)   # (B, H)
    lidar_hi = lidar.gather(1, idx_hi.view(B, -1)).view(B, H)   # (B, H)

    # Linear interpolation: wall distance in waypoint direction
    d_wall = lidar_lo * (1.0 - alpha) + lidar_hi * alpha         # (B, H)

    # Safe margin
    margin = (d_wall - robot_radius).clamp(min=eps)               # (B, H)

    if penalty_type == "exp":
        # clearance > 0: safe; clearance < 0: inside wall
        clearance = margin - wp_dist                              # (B, H)
        # clamp to avoid exp exploding deep inside walls
        penalty = torch.exp(-clearance.clamp(max=5.0 * sigma) / sigma)
    else:
        # relu: only penalize waypoints past the wall
        penalty = F.relu(wp_dist - margin) ** 2

    return penalty.mean()


@torch.no_grad()
def check_collision(
    waypoints_xy: torch.Tensor,  # (B, H, 2) local-frame positions (metres)
    lidar: torch.Tensor,         # (B, n_beams) raw metres
    robot_radius: float = 0.15,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Binary collision check for each trajectory in the batch.

    Returns (B,) bool tensor — True if any waypoint falls inside the wall
    (i.e. wp_dist > d_wall - robot_radius).
    """
    B, H, _ = waypoints_xy.shape
    n_beams = lidar.shape[1]

    dx = waypoints_xy[..., 0]
    dy = waypoints_xy[..., 1]
    wp_dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    angle = torch.atan2(dy, dx) % (2.0 * math.pi)
    frac_idx = angle / (2.0 * math.pi) * n_beams

    idx_lo = frac_idx.long() % n_beams
    idx_hi = (idx_lo + 1) % n_beams
    alpha = frac_idx - frac_idx.detach().floor()

    lidar_lo = lidar.gather(1, idx_lo.view(B, -1)).view(B, H)
    lidar_hi = lidar.gather(1, idx_hi.view(B, -1)).view(B, H)
    d_wall = lidar_lo * (1.0 - alpha) + lidar_hi * alpha

    margin = (d_wall - robot_radius).clamp(min=eps)
    return (wp_dist > margin).any(dim=-1)   # (B,) bool


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
        lidar_max_range: float = 3.5,
    ) -> Dict[str, torch.Tensor]:
        """
        OT-CFM training loss with optional differentiable collision penalty.

        non_lidar:      (B, non_lidar_dim)
        lidar:          (B, n_beams)  raw metres (normalize_lidar=False) or [0,1]
        target_flat:    (B, H*4)  clean trajectory x_1
        lidar_max_range: used to compute real-metre wall distances for collision check
        returns dict with 'loss', 'flow_loss', and optionally 'collision_loss'
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
        if self.cfg.sigma_min > 0:
            x_t = x_t + self.cfg.sigma_min * torch.randn_like(x_t)

        # OT-CFM target velocity (constant along the linear path)
        v_target = x_1 - x_0                                # (B, H, 4)

        # Encode conditioning
        global_cond = self.encode_condition(non_lidar, lidar)

        # Predict velocity
        v_pred = self.predict_velocity(x_t, t, global_cond)  # (B, H, 4)

        flow_loss = F.mse_loss(v_pred, v_target)

        if self.cfg.collision_loss_weight <= 0.0:
            return {"loss": flow_loss, "flow_loss": flow_loss}

        # Differentiable collision penalty on the predicted clean trajectory.
        # x_1_pred = x_t + (1-t)*v_pred  recovers the predicted x_1 from the
        # current interpolated point and the velocity, giving a differentiable
        # path from model parameters to waypoint positions.
        x_1_pred = x_t + (1.0 - t_bcast) * v_pred           # (B, H, action_dim)
        waypoints_xy = x_1_pred[:, :, :2]                    # (B, H, 2) local metres

        # Convert lidar to metres if it was stored normalised (legacy datasets).
        # For new datasets (normalize_lidar=False) lidar is already in metres.
        # We check the scale: if max > 2.0 it's almost certainly metres already.
        lidar_m = lidar if lidar.max().item() > 2.0 else lidar * lidar_max_range

        coll_loss = compute_collision_loss(
            waypoints_xy, lidar_m, lidar_max_range,
            robot_radius=self.cfg.robot_radius,
            penalty_type=self.cfg.collision_penalty_type,
            sigma=self.cfg.collision_sigma,
        )

        total_loss = flow_loss + self.cfg.collision_loss_weight * coll_loss
        return {"loss": total_loss, "flow_loss": flow_loss, "collision_loss": coll_loss}

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
    collision_samples: int = 0,
    lidar_max_range: float = 3.5,
    robot_radius: float = 0.15,
) -> Dict[str, float]:
    """
    Evaluate on the validation set.

    For each batch, draws `num_samples` independent trajectory samples and computes:
      loss              — MSE training proxy (fast, single forward pass)
      ade / fde         — mean ADE/FDE over all samples
      min_ade           — best-of-N ADE  (primary metric for generative models)
      min_fde           — best-of-N FDE
      diversity         — mean pairwise L2 between samples (mode-collapse detector)
      heading_err       — mean angular error of the best sample (radians)
      collision_rate    — fraction of samples with ≥1 colliding waypoint  (if collision_samples>0)
      scene_coll_rate   — fraction of scenes where ALL samples collide     (if collision_samples>0)
    """
    model.eval()
    accum: Dict[str, float] = {
        "loss": 0.0, "ade": 0.0, "fde": 0.0,
        "min_ade": 0.0, "min_fde": 0.0,
        "diversity": 0.0, "heading_err": 0.0,
        "collision_rate": 0.0, "scene_coll_rate": 0.0,
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

        # Collision rate: draw collision_samples trajectories and check each one
        if collision_samples > 0:
            lidar_m = lidar if lidar.max().item() > 2.0 else lidar * lidar_max_range
            coll_flags = torch.stack(
                [
                    check_collision(
                        model.sample_trajectory(non_lidar, lidar, num_inference_steps=num_inference_steps)[:, :, :2],
                        lidar_m, robot_radius=robot_radius,
                    ).float()
                    for _ in range(collision_samples)
                ],
                dim=1,
            )  # (B, collision_samples)
            accum["collision_rate"] += float(coll_flags.mean().item())
            accum["scene_coll_rate"] += float(coll_flags.all(dim=1).float().mean().item())

        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in accum.items()}


__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingPolicyModel",
    "compute_collision_loss",
    "check_collision",
    "evaluate_epoch",
    "LocalTrajDataset",
    "traj_metrics",
]
