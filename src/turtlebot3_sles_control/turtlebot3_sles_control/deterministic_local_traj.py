#!/usr/bin/env python3
"""
Deterministic baseline model for local trajectory prediction.

This module defines (same schema as create_dataset_imitation_local_traj outputs):
  - LocalTrajDataset: dataset loader for (inputs, outputs, metadata)
  - LidarCNN1D: 1D CNN encoder for lidar scans
  - traj_metrics: ADE / FDE metrics on (dx, dy)

Model:
  - Separate lidar encoder (1D CNN) -> lidar embedding
  - Concatenate non-lidar features + lidar embedding -> global condition
  - Temporal decoder: small Transformer that operates on (B, H, 4)
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LocalTrajDataset(Dataset):
    """
    Loads flattened inputs/outputs and splits lidar from non-lidar features.
    """

    def __init__(self, npz_file: str, eps: float = 1e-6):
        data = np.load(npz_file, allow_pickle=True)
        if "inputs" not in data.files or "outputs" not in data.files:
            raise KeyError("Expected keys 'inputs' and 'outputs' in dataset npz")

        self.inputs = data["inputs"].astype(np.float32)
        self.outputs = data["outputs"].astype(np.float32)

        self.horizon = int(data["horizon"]) if "horizon" in data.files else (self.outputs.shape[1] // 4)
        self.target_dim = self.outputs.shape[1]

        self.n_beams = int(data["n_beams"]) if "n_beams" in data.files else None
        if self.n_beams is None:
            raise KeyError("Dataset missing 'n_beams' metadata (needed to split lidar from inputs).")

        if self.inputs.ndim != 2:
            raise ValueError("inputs must be 2D (N, input_dim)")
        if self.outputs.ndim != 2:
            raise ValueError("outputs must be 2D (N, horizon*4)")
        if self.outputs.shape[1] != self.horizon * 4:
            raise ValueError("outputs dim mismatch with horizon*4")

        if self.inputs.shape[1] < self.n_beams:
            raise ValueError("input_dim < n_beams; cannot split lidar")

        self.non_lidar_dim = self.inputs.shape[1] - self.n_beams
        if self.non_lidar_dim <= 0:
            raise ValueError("non_lidar_dim <= 0; check dataset schema")

        outs = self.outputs.reshape(self.outputs.shape[0], self.horizon, 4)
        unit = outs[..., 2] ** 2 + outs[..., 3] ** 2
        max_err = float(np.max(np.abs(unit - 1.0)))
        if not np.isfinite(max_err) or max_err > 1e-3:
            raise ValueError("Dataset outputs failed sin/cos unit check; regenerate dataset.")

        self.meta = {k: data[k] for k in data.files if k not in ("inputs", "outputs")}
        self.eps = eps

        print(f"Loaded dataset {npz_file}")
        print(f"  inputs:  {self.inputs.shape} (non_lidar_dim={self.non_lidar_dim}, n_beams={self.n_beams})")
        print(f"  outputs: {self.outputs.shape} (horizon={self.horizon})")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]
        y = self.outputs[idx]
        non_lidar = torch.from_numpy(x[: self.non_lidar_dim])
        lidar = torch.from_numpy(x[self.non_lidar_dim :])
        target = torch.from_numpy(y)
        return non_lidar.float(), lidar.float(), target.float()


class LidarCNN1D(nn.Module):
    """
    Simple 1D CNN encoder for lidar scan (B, n_beams) -> (B, lidar_emb_dim).
    """

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
        x = lidar.unsqueeze(1)
        h = self.net(x)
        return self.proj(h)


def traj_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    horizon: int,
    best_of_n: bool = False,
) -> Dict[str, float]:
    """
    Compute MSE, ADE, FDE on (dx, dy) from predicted trajectory vector.

    pred:
      - if best_of_n=False: (B, target_dim)
      - if best_of_n=True:  (B, N, target_dim)
    gt: (B, target_dim)
    """
    with torch.no_grad():
        if not best_of_n:
            mse = torch.mean((pred - gt) ** 2).item()
            p = pred.view(pred.shape[0], horizon, 4)[..., :2]
            g = gt.view(gt.shape[0], horizon, 4)[..., :2]
            d = torch.norm(p - g, dim=-1)
            ade = torch.mean(d).item()
            fde = torch.mean(d[:, -1]).item()
            return {"mse": mse, "ade": ade, "fde": fde}

        b, n, td = pred.shape
        p = pred.view(b, n, horizon, 4)[..., :2]
        g = gt.view(b, 1, horizon, 4)[..., :2]
        d = torch.norm(p - g, dim=-1)
        ade_n = torch.mean(d, dim=-1)
        best = torch.argmin(ade_n, dim=1)
        idx = best.view(b, 1, 1).expand(b, 1, horizon)
        d_best = torch.gather(d, dim=1, index=idx).squeeze(1)

        ade = torch.mean(d_best).item()
        fde = torch.mean(d_best[:, -1]).item()
        best_vec = pred[torch.arange(b, device=pred.device), best, :]
        mse = torch.mean((best_vec - gt) ** 2).item()
        return {"mse": mse, "ade": ade, "fde": fde}


@dataclass
class DeterministicConfig:
    horizon: int
    non_lidar_dim: int
    n_beams: int
    lidar_emb_dim: int = 64
    cond_dim: int = 256
    model_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


class TemporalTrajectoryTransformer(nn.Module):
    """
    Simple Transformer encoder that maps a sequence of per-step embeddings
    to per-step trajectory outputs (dx, dy, sin(dtheta), cos(dtheta)).

    Input:  (B, H, D)
    Output: (B, H, 4)
    """

    def __init__(self, horizon: int, d_model: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.num_layers = int(num_layers)

        # Learned positional embeddings over horizon steps
        self.pos_emb = nn.Embedding(horizon, d_model)

        # Use explicit layers so we can inject conditioning after every layer.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=False,  # (S, B, E)
                    activation="relu",
                    norm_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.head = nn.Linear(d_model, 4)

    def forward(self, cond_seq: torch.Tensor) -> torch.Tensor:
        """
        Residual conditioning (Option A):
          - initialize x from positional embeddings only
          - after each transformer layer: x = x + cond_seq

        cond_seq: (B, H, D) broadcast condition sequence
        returns traj: (B, H, 4)
        """
        b, h, d = cond_seq.shape
        assert h == self.horizon, f"expected horizon={self.horizon}, got {h}"
        assert d == self.d_model, f"expected d_model={self.d_model}, got {d}"

        idx = torch.arange(h, device=cond_seq.device)
        pos = self.pos_emb(idx).unsqueeze(0).expand(b, h, -1)  # (B, H, D)
        x = pos  # (B, H, D)

        x = x.permute(1, 0, 2)  # (H, B, D)
        c = cond_seq.permute(1, 0, 2)  # (H, B, D)
        for layer in self.layers:
            x = layer(x)  # (H, B, D)
            x = x + c

        x = x.permute(1, 0, 2).contiguous()  # (B, H, D)
        traj = self.head(x)  # (B, H, 4)
        return traj


class DeterministicLocalTrajModel(nn.Module):
    """
    Deterministic baseline:
      (current state + history + commands + target + lidar) -> future (B, H, 4).

    - Lidar encoder: LidarCNN1D -> (B, lidar_emb_dim)
    - Condition encoder: MLP(non_lidar + lidar_emb) -> (B, cond_dim)
    - Temporal decoder: small Transformer that operates over horizon steps,
      conditioned by broadcasting the global condition to each time step.
    """

    def __init__(self, cfg: DeterministicConfig):
        super().__init__()
        self.cfg = cfg
        self.horizon = int(cfg.horizon)

        self.lidar_enc = LidarCNN1D(n_beams=cfg.n_beams, emb_dim=cfg.lidar_emb_dim)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cfg.non_lidar_dim + cfg.lidar_emb_dim, cfg.cond_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        self.cond_to_model = nn.Linear(cfg.cond_dim, cfg.model_dim)

        self.temporal = TemporalTrajectoryTransformer(
            horizon=self.horizon,
            d_model=cfg.model_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )

    def forward(self, non_lidar: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """
        non_lidar: (B, non_lidar_dim)
        lidar:     (B, n_beams)
        returns:   (B, H, 4)
        """
        lidar_emb = self.lidar_enc(lidar)  # (B, lidar_emb_dim)
        cond = self.cond_mlp(torch.cat([non_lidar, lidar_emb], dim=1))  # (B, cond_dim)
        base = self.cond_to_model(cond)  # (B, model_dim)

        # Broadcast global condition to each horizon step
        b = base.shape[0]
        h = self.horizon
        cond_seq = base.unsqueeze(1).expand(b, h, -1)  # (B, H, model_dim)

        traj = self.temporal(cond_seq)  # (B, H, 4)
        return traj


def trajectory_loss(
    pred_traj: torch.Tensor,
    target_flat: torch.Tensor,
    horizon: int,
    lambda_smooth: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Compute trajectory MSE + smoothness loss on (dx, dy, sin(dtheta), cos(dtheta)).

    pred_traj: (B, H, 4)
    target_flat: (B, H*4)
    """
    b, h, _ = pred_traj.shape
    assert h == horizon, f"expected horizon={horizon}, got {h}"

    target = target_flat.view(b, horizon, 4)

    mse = torch.mean((pred_traj - target) ** 2)

    # Smoothness via second-difference over time.
    # Apply to all channels, including heading representation (sin/cos), to avoid angle-wrap issues.
    d2 = pred_traj[:, 2:, :] - 2.0 * pred_traj[:, 1:-1, :] + pred_traj[:, :-2, :]  # (B, H-2, 4)
    smooth = torch.mean(d2 * d2)

    total = mse + lambda_smooth * smooth
    return {"total": total, "traj_mse": mse, "smooth": smooth}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    horizon: int,
) -> Dict[str, float]:
    """
    Compute average loss and ADE/FDE on a dataset.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_smooth = 0.0
    total_ade = 0.0
    total_fde = 0.0
    n_batches = 0

    for non_lidar, lidar, target in loader:
        non_lidar = non_lidar.to(device)
        lidar = lidar.to(device)
        target = target.to(device)

        pred_traj = model(non_lidar, lidar)  # (B, H, 4)
        losses = trajectory_loss(pred_traj, target, horizon=horizon)

        total_loss += float(losses["total"].item())
        total_mse += float(losses["traj_mse"].item())
        total_smooth += float(losses["smooth"].item())

        # Metrics expect flattened predictions
        pred_flat = pred_traj.reshape(pred_traj.shape[0], -1)
        m = traj_metrics(pred_flat, target, horizon=horizon, best_of_n=False)
        total_ade += float(m["ade"])
        total_fde += float(m["fde"])

        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "traj_mse": total_mse / denom,
        "smooth": total_smooth / denom,
        "ade": total_ade / denom,
        "fde": total_fde / denom,
    }


__all__ = [
    "DeterministicConfig",
    "DeterministicLocalTrajModel",
    "trajectory_loss",
    "evaluate_epoch",
    "LocalTrajDataset",
    "traj_metrics",
]

