#!/usr/bin/env python3
"""
Live policy runner — wraps a trained Diffusion / Flow / Deterministic model for
real-time use in a simulator or on a robot.

The runner maintains rolling state and command history buffers.  On each call to
`get_action()` you pass the current observation and get back a predicted trajectory
(in the robot's current frame) without touching any .npz files.

Coordinate conventions (match dataset creation):
  Robot frame: +x = forward, +y = left.
  theta: CCW-positive, world-frame heading.
  Trajectory outputs (dx_k, dy_k, sin_dtheta_k, cos_dtheta_k):
    dx_k, dy_k  — position of waypoint k relative to robot pose at t=0, IN robot frame.
    dtheta_k    — heading change from t=0 to waypoint k.
  i.e. outputs are NOT incremental step-to-step deltas; they are all relative to NOW.

Quick start
-----------
    from policy_runner import PolicyRunner, Observation

    runner = PolicyRunner("models/diffusion_policy/best_model_diffusion_policy.pth")

    # In your simulator loop:
    obs = Observation(
        x=1.23, y=4.56, theta=0.78,          # world frame pose
        v=0.1, w=0.05,                        # current velocity
        v_cmd=0.12, w_cmd=0.03,               # most recent command sent
        target_x=5.0, target_y=6.0,          # goal in world frame
        lidar_ranges=np.array([...]),         # (n_beams,) in metres
        lidar_max_range=3.5,                  # used to normalise to [0,1]
        timestamp=t,                          # current time in seconds
    )
    traj = runner.get_trajectory(obs)
    # traj: (horizon, 4) — (dx, dy, sin_dtheta, cos_dtheta) in robot frame
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Normalisation constants — must match the dataset you trained on.
# Override via PolicyRunner(..., v_max=..., w_max=...) if needed.
# ---------------------------------------------------------------------------
DEFAULT_V_MAX = 0.26   # m/s
DEFAULT_W_MAX = 1.5    # rad/s
DEFAULT_DT    = 0.1    # seconds (history / horizon step size)


# ---------------------------------------------------------------------------
# Observation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    Single time-step observation from the simulator / robot.

    All poses are in the world frame.
    lidar_ranges: raw range values in metres (invalid/no-return → float('inf') or ≥ lidar_max_range).
    lidar_max_range: used to normalise ranges to [0,1].  Ranges ≥ max_range become 1.0.
    timestamp: wall-clock or sim time in seconds.  Used to align history lookups.
    """
    x: float
    y: float
    theta: float          # world-frame heading, radians, CCW-positive
    v: float              # linear velocity  (m/s)
    w: float              # angular velocity (rad/s)
    v_cmd: float          # linear velocity command sent at this step
    w_cmd: float          # angular velocity command sent at this step
    target_x: float       # goal x in world frame
    target_y: float       # goal y in world frame
    lidar_ranges: np.ndarray   # (n_beams,) metres
    lidar_max_range: float = 3.5
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _world_to_robot(dx_w: float, dy_w: float, theta: float) -> Tuple[float, float]:
    """Rotate a world-frame delta into the robot frame (SE(2) rotation only)."""
    c, s = math.cos(theta), math.sin(theta)
    return dx_w * c + dy_w * s, -dx_w * s + dy_w * c


def _normalise_lidar(ranges: np.ndarray, max_range: float) -> np.ndarray:
    """
    Normalise raw lidar ranges to [0, 1].
      - ranges ≥ max_range (or inf/nan)  →  1.0  (no return)
      - ranges < 0                        →  1.0  (invalid)
      - otherwise                         →  range / max_range
    """
    out = np.asarray(ranges, dtype=np.float32).copy()
    out = np.where(np.isfinite(out), out, max_range)   # inf/nan → max_range
    out = np.where(out < 0, max_range, out)             # negative → max_range
    out = np.clip(out / max_range, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# PolicyRunner
# ---------------------------------------------------------------------------

class PolicyRunner:
    """
    Maintains history buffers and drives inference for a single policy model.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to a .pth checkpoint saved by train_diffusion_policy.py,
        train_flow_matching.py, or train_deterministic_local_traj.py.
    model_type : 'diffusion' | 'flow' | 'deterministic' | None
        Auto-detected from filename if not given.
    device : str
        'cuda' or 'cpu'.
    num_inference_steps : int
        Denoising / ODE steps.  10 is fast; 100 is full quality.
    num_samples : int
        Trajectories to draw per call (generative models).  The one with lowest
        predicted position error vs the others' mean is returned as the "best" sample.
        Set to 1 to always return the single sample (faster).
    v_max, w_max : float
        Velocity normalisation constants — must match what was used during training.
    dt : float
        Time step between history entries — must match dataset dt.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        num_inference_steps: int = 10,
        num_samples: int = 1,
        v_max: float = DEFAULT_V_MAX,
        w_max: float = DEFAULT_W_MAX,
        dt: float = DEFAULT_DT,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.num_samples = num_samples
        self.v_max = v_max
        self.w_max = w_max
        self.dt = dt

        self.model, self.model_type, self.cfg, self.meta = _load_model(
            str(ckpt_path), device, model_type
        )

        def _cfg_int(key: str, default: int) -> int:
            """cfg first (skip None), then meta (convert ndarray), then default."""
            v = self.cfg.get(key)
            if v is None:
                raw = self.meta.get(key) if self.meta else None
                v = int(np.asarray(raw).flat[0]) if raw is not None else None
            return int(v) if v is not None else default

        def _cfg_float(key: str, default: float) -> float:
            v = self.cfg.get(key)
            if v is None:
                raw = self.meta.get(key) if self.meta else None
                v = float(np.asarray(raw).flat[0]) if raw is not None else None
            return float(v) if v is not None else default

        self.history_len: int = _cfg_int("history_len", 10)
        self.horizon: int     = _cfg_int("horizon", 20)
        # v_max/w_max must match what was used during dataset creation
        self.v_max = _cfg_float("v_max", v_max)
        self.w_max = _cfg_float("w_max", w_max)
        self.include_heading_diff: bool = bool(self.cfg.get("include_heading_diff", False))

        # Rolling history buffers — each entry is a tuple stored at push time
        # state_history: deque of (x, y, theta, v, w, timestamp)
        # cmd_history:   deque of (v_cmd, w_cmd, timestamp)
        maxlen = self.history_len + 20  # keep extra for flexible lookup
        self._state_buf: deque = deque(maxlen=maxlen)
        self._cmd_buf:   deque = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear history buffers (call at the start of each episode)."""
        self._state_buf.clear()
        self._cmd_buf.clear()

    def update(self, obs: Observation) -> None:
        """
        Push one observation into the history buffers.

        Call this BEFORE get_trajectory() if you want the current obs included
        in the history, or call get_trajectory(obs) directly which does both.
        """
        self._state_buf.append((obs.x, obs.y, obs.theta, obs.v, obs.w, obs.timestamp))
        self._cmd_buf.append((obs.v_cmd, obs.w_cmd, obs.timestamp))

    @torch.no_grad()
    def get_trajectory(self, obs: Observation) -> np.ndarray:
        """
        Build the input vector from `obs` + history, run the model, and return
        the predicted trajectory.

        Parameters
        ----------
        obs : Observation
            Current observation.  Also pushed into the history buffer.

        Returns
        -------
        traj : np.ndarray  shape (horizon, 4)
            Predicted waypoints (dx, dy, sin_dtheta, cos_dtheta) in the current
            robot frame.  Each row is relative to the robot's pose RIGHT NOW
            (not step-to-step incremental).

            To convert waypoint k to world frame:
                x_k = obs.x + dx_k * cos(obs.theta) - dy_k * sin(obs.theta)
                y_k = obs.y + dx_k * sin(obs.theta) + dy_k * cos(obs.theta)
                theta_k = obs.theta + atan2(sin_dtheta_k, cos_dtheta_k)
        """
        self.update(obs)
        non_lidar, lidar = self._build_input(obs)
        non_lidar_t = torch.from_numpy(non_lidar).unsqueeze(0).to(self.device)
        lidar_t     = torch.from_numpy(lidar).unsqueeze(0).to(self.device)

        if self.model_type == "deterministic":
            traj = self.model(non_lidar_t, lidar_t).squeeze(0)  # (H, 4)
        else:
            if self.num_samples == 1:
                traj = self.model.sample_trajectory(
                    non_lidar_t, lidar_t,
                    num_inference_steps=self.num_inference_steps,
                ).squeeze(0)  # (H, 4)
            else:
                samples = torch.stack([
                    self.model.sample_trajectory(
                        non_lidar_t, lidar_t,
                        num_inference_steps=self.num_inference_steps,
                    ).squeeze(0)
                    for _ in range(self.num_samples)
                ])  # (num_samples, H, 4)
                # Pick sample closest to the mean (avoids outlier draws)
                mean_xy = samples[..., :2].mean(dim=0, keepdim=True)  # (1, H, 2)
                dists = (samples[..., :2] - mean_xy).norm(dim=-1).mean(dim=-1)  # (num_samples,)
                best  = dists.argmin()
                traj  = samples[best]  # (H, 4)

        return traj.cpu().numpy()

    @torch.no_grad()
    def get_all_samples(self, obs: Observation) -> np.ndarray:
        """
        Same as get_trajectory but returns ALL sampled trajectories.

        Returns
        -------
        trajs : np.ndarray  shape (num_samples, horizon, 4)
        """
        self.update(obs)
        non_lidar, lidar = self._build_input(obs)
        non_lidar_t = torch.from_numpy(non_lidar).unsqueeze(0).to(self.device)
        lidar_t     = torch.from_numpy(lidar).unsqueeze(0).to(self.device)

        if self.model_type == "deterministic":
            traj = self.model(non_lidar_t, lidar_t).squeeze(0).cpu().numpy()
            return traj[None]  # (1, H, 4)

        samples = np.stack([
            self.model.sample_trajectory(
                non_lidar_t, lidar_t,
                num_inference_steps=self.num_inference_steps,
            ).squeeze(0).cpu().numpy()
            for _ in range(self.num_samples)
        ])
        return samples  # (num_samples, H, 4)

    # ------------------------------------------------------------------
    # Input construction  (mirrors create_samples_from_session exactly)
    # ------------------------------------------------------------------

    def _lookup_state(self, t: float):
        """
        Return the state entry with the largest timestamp <= t.
        Falls back to the oldest entry if none qualify.
        """
        best = None
        for entry in self._state_buf:
            ts = entry[5]
            if ts <= t:
                best = entry
        return best or (self._state_buf[0] if self._state_buf else None)

    def _lookup_cmd(self, t: float):
        """Return (v_cmd, w_cmd) at largest timestamp <= t, or (0, 0) if missing."""
        best = None
        for entry in self._cmd_buf:
            ts = entry[2]
            if ts <= t:
                best = entry
        if best is None:
            return 0.0, 0.0
        return best[0], best[1]

    def _build_input(self, obs: Observation) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct (non_lidar, lidar) from the current observation + history buffers.

        Layout matches create_dataset_imitation_local_traj.py:
          non_lidar = [curr(6) | state_hist(H*6) | cmd_hist(H*2) | goal(2) | heading(0 or 2)]
          lidar     = normalised scan (n_beams,)
        """
        t      = obs.timestamp
        x_t    = obs.x
        y_t    = obs.y
        theta_t = obs.theta
        v_t    = obs.v / self.v_max
        w_t    = obs.w / self.w_max
        sin_t  = math.sin(theta_t)
        cos_t  = math.cos(theta_t)

        # Goal in robot frame
        dx_g, dy_g = _world_to_robot(obs.target_x - x_t, obs.target_y - y_t, theta_t)

        curr_part = np.array([v_t, w_t], dtype=np.float32)

        # State history (oldest → newest), H entries — pre-transformed to local robot frame at t
        cos_t = math.cos(theta_t)
        state_hist: list[float] = []
        for j in range(self.history_len):
            k = self.history_len - j          # k=H for oldest, k=1 for most recent
            query = t - k * self.dt
            entry = self._lookup_state(query)
            if entry is None:
                # No history: repeat current state (which is at origin in local frame)
                state_hist.extend([0.0, 0.0, 0.0, 1.0, v_t, w_t])
            else:
                xk, yk, thk, vk, wk, _ = entry
                dx_w, dy_w = xk - x_t, yk - y_t
                dx_l =  dx_w * cos_t + dy_w * sin_t
                dy_l = -dx_w * sin_t + dy_w * cos_t
                dtheta = (thk - theta_t + math.pi) % (2.0 * math.pi) - math.pi
                state_hist.extend([
                    dx_l, dy_l,
                    math.sin(dtheta), math.cos(dtheta),
                    vk / self.v_max, wk / self.w_max,
                ])

        # Command history (oldest → newest), H entries
        cmd_hist: list[float] = []
        for j in range(self.history_len):
            k = self.history_len - j
            query = t - k * self.dt
            vc, wc = self._lookup_cmd(query)
            cmd_hist.extend([vc / self.v_max, wc / self.w_max])

        goal_part = np.array([dx_g, dy_g], dtype=np.float32)

        heading_part: list[float] = []
        if self.include_heading_diff:
            r = math.hypot(dx_g, dy_g)
            if r < 1e-9:
                heading_part = [0.0, 1.0]
            else:
                heading_part = [dy_g / r, dx_g / r]  # (sin, cos) of angle to goal in robot frame

        non_lidar = np.concatenate([
            curr_part,
            np.array(state_hist, dtype=np.float32),
            np.array(cmd_hist,   dtype=np.float32),
            goal_part,
            np.array(heading_part, dtype=np.float32),
        ]).astype(np.float32)

        lidar = _normalise_lidar(obs.lidar_ranges, obs.lidar_max_range)

        return non_lidar, lidar

    # ------------------------------------------------------------------
    # Utility: convert trajectory to world frame
    # ------------------------------------------------------------------

    @staticmethod
    def traj_to_world(traj: np.ndarray, x0: float, y0: float, theta0: float) -> np.ndarray:
        """
        Convert a (horizon, 4) robot-frame trajectory to world-frame poses.

        Parameters
        ----------
        traj   : (H, 4) — (dx, dy, sin_dtheta, cos_dtheta) in robot frame
        x0, y0, theta0 : robot pose at the time the trajectory was predicted

        Returns
        -------
        poses  : (H, 3) — (x_world, y_world, theta_world)
        """
        H = traj.shape[0]
        poses = np.zeros((H, 3), dtype=np.float32)
        c0, s0 = math.cos(theta0), math.sin(theta0)
        for k in range(H):
            dx_l, dy_l = traj[k, 0], traj[k, 1]
            # Rotate local → world
            dx_w = dx_l * c0 - dy_l * s0
            dy_w = dx_l * s0 + dy_l * c0
            poses[k, 0] = x0 + dx_w
            poses[k, 1] = y0 + dy_w
            poses[k, 2] = theta0 + math.atan2(float(traj[k, 2]), float(traj[k, 3]))
        return poses


# ---------------------------------------------------------------------------
# Internal: model loading (reuses logic from infer.py)
# ---------------------------------------------------------------------------

def _detect_model_type(ckpt_path: str) -> str:
    name = Path(ckpt_path).stem.lower()
    if "flow" in name:
        return "flow"
    if "deterministic" in name:
        return "deterministic"
    return "diffusion"


def _load_model(ckpt_path: str, device: str, model_type: Optional[str] = None):
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    meta     = ckpt.get("dataset_meta", {})

    if model_type is None:
        model_type = _detect_model_type(ckpt_path)

    def _get(key, default):
        """cfg_dict first, then meta (scalar), then default. Handles stored None."""
        v = cfg_dict.get(key)
        if v is None:
            raw = meta.get(key)
            v = int(raw) if raw is not None else None
        return v if v is not None else default

    if model_type == "diffusion":
        from diffusion_policy_model import DiffusionPolicyConfig, DiffusionPolicyModel
        cfg = DiffusionPolicyConfig(
            horizon=cfg_dict["horizon"],
            non_lidar_dim=cfg_dict["non_lidar_dim"],
            n_beams=cfg_dict["n_beams"],
            history_len=_get("history_len", 10),
            include_heading_diff=cfg_dict.get("include_heading_diff", False),
            action_dim=_get("action_dim", 4),
            lidar_emb_dim=cfg_dict.get("lidar_emb_dim", 64),
            global_cond_dim=cfg_dict.get("global_cond_dim", 256),
            diffusion_step_embed_dim=cfg_dict.get("diffusion_step_embed_dim", 128),
            down_dims=tuple(cfg_dict.get("down_dims", [256, 512, 1024])),
            kernel_size=cfg_dict.get("kernel_size", 3),
            n_groups=cfg_dict.get("n_groups", 8),
            cond_predict_scale=cfg_dict.get("cond_predict_scale", False),
            num_train_timesteps=cfg_dict.get("num_train_timesteps", 100),
            num_inference_steps=cfg_dict.get("num_inference_steps", 100),
            beta_schedule=cfg_dict.get("beta_schedule", "squaredcos_cap_v2"),
            prediction_type=cfg_dict.get("prediction_type", "epsilon"),
        )
        model = DiffusionPolicyModel(cfg).to(device)

    elif model_type == "flow":
        from flow_matching_model import FlowMatchingConfig, FlowMatchingPolicyModel
        cfg = FlowMatchingConfig(
            horizon=cfg_dict["horizon"],
            non_lidar_dim=cfg_dict["non_lidar_dim"],
            n_beams=cfg_dict["n_beams"],
            history_len=_get("history_len", 10),
            include_heading_diff=cfg_dict.get("include_heading_diff", False),
            action_dim=_get("action_dim", 4),
            lidar_emb_dim=cfg_dict.get("lidar_emb_dim", 64),
            global_cond_dim=cfg_dict.get("global_cond_dim", 256),
            diffusion_step_embed_dim=cfg_dict.get("diffusion_step_embed_dim", 128),
            down_dims=tuple(cfg_dict.get("down_dims", [256, 512, 1024])),
            kernel_size=cfg_dict.get("kernel_size", 3),
            n_groups=cfg_dict.get("n_groups", 8),
            cond_predict_scale=cfg_dict.get("cond_predict_scale", False),
            num_inference_steps=cfg_dict.get("num_inference_steps", 100),
            integrator=cfg_dict.get("integrator", "euler"),
        )
        model = FlowMatchingPolicyModel(cfg).to(device)

    elif model_type == "deterministic":
        from deterministic_local_traj import DeterministicConfig, DeterministicLocalTrajModel
        cfg = DeterministicConfig(
            horizon=cfg_dict["horizon"],
            non_lidar_dim=cfg_dict["non_lidar_dim"],
            n_beams=cfg_dict["n_beams"],
            history_len=cfg_dict.get("history_len", 10),
            include_heading_diff=cfg_dict.get("include_heading_diff", False),
            lidar_emb_dim=cfg_dict.get("lidar_emb_dim", 64),
            cond_dim=cfg_dict.get("cond_dim", 256),
            model_dim=cfg_dict.get("model_dim", 128),
            num_layers=cfg_dict.get("num_layers", 2),
            num_heads=cfg_dict.get("num_heads", 4),
            dropout=cfg_dict.get("dropout", 0.0),
        )
        model = DeterministicLocalTrajModel(cfg).to(device)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[PolicyRunner] loaded {model_type} model from {ckpt_path}", flush=True)
    return model, model_type, cfg_dict, meta


# ---------------------------------------------------------------------------
# Minimal example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Smoke-test PolicyRunner with random observations.")
    parser.add_argument("--ckpt",    required=True, help="Checkpoint .pth file")
    parser.add_argument("--n-beams", type=int, default=360, help="Lidar beam count (must match training)")
    parser.add_argument("--steps",   type=int, default=5,   help="How many fake timesteps to run")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    args = parser.parse_args()

    runner = PolicyRunner(
        args.ckpt,
        num_inference_steps=args.num_inference_steps,
        num_samples=args.num_samples,
    )
    runner.reset()

    rng = np.random.default_rng(0)
    for step in range(args.steps):
        t = step * 0.1
        obs = Observation(
            x=rng.uniform(-1, 1), y=rng.uniform(-1, 1), theta=rng.uniform(-np.pi, np.pi),
            v=rng.uniform(0, 0.2), w=rng.uniform(-0.5, 0.5),
            v_cmd=0.1, w_cmd=0.0,
            target_x=3.0, target_y=0.0,
            lidar_ranges=rng.uniform(0.5, 3.5, size=args.n_beams),
            lidar_max_range=3.5,
            timestamp=t,
        )
        traj = runner.get_trajectory(obs)
        world = PolicyRunner.traj_to_world(traj, obs.x, obs.y, obs.theta)
        print(f"step {step}: first waypoint local=({traj[0,0]:.3f}, {traj[0,1]:.3f})  "
              f"world=({world[0,0]:.3f}, {world[0,1]:.3f})")

    print("Done.")
