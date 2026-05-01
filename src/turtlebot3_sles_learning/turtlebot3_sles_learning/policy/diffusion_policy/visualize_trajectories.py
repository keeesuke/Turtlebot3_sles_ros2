#!/usr/bin/env python3
"""
Top-down trajectory visualizer for offline sanity checking.

For each sample, renders a bird's-eye view in the robot frame:
  - Lidar point cloud (gray)
  - GT trajectory (blue, solid)
  - Predicted trajectory samples (red, semi-transparent) — optional, requires --checkpoint

Everything is in the canonical robot frame: origin = robot position, x-axis = forward.
No world-frame conversion needed since lidar and trajectories are already in this frame.

Usage (dataset only — GT sanity check):
  python visualize_trajectories.py --dataset val.npz --n-samples 16 --out-dir vis/

Usage (with model predictions):
  python visualize_trajectories.py --dataset val.npz --checkpoint model.pth \
      --n-pred-samples 8 --n-samples 16 --out-dir vis/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lidar projection
# ---------------------------------------------------------------------------

def lidar_to_xy(scan: np.ndarray, n_beams: int) -> tuple:
    """
    Convert a 1D range scan to Cartesian in robot frame.
    Beams are evenly spaced from -pi to pi.

    Returns:
      hit_x, hit_y   : valid hit positions (solid dots)
      miss_x, miss_y : max-range / invalid beam endpoints (faint ring dots)
    """
    angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
    valid = scan < 1.0           # exactly 1.0 = invalid / max range sentinel
    r_hit  = scan[valid]
    r_miss = scan[~valid]        # all at exactly 1.0 (the sentinel value)
    a_hit  = angles[valid]
    a_miss = angles[~valid]
    hit_x  = r_hit  * np.cos(a_hit)
    hit_y  = r_hit  * np.sin(a_hit)
    miss_x = r_miss * np.cos(a_miss)   # r_miss == 1.0, so these sit on the unit circle
    miss_y = r_miss * np.sin(a_miss)
    return hit_x, hit_y, miss_x, miss_y


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_sample(
    lidar: np.ndarray,           # (n_beams,) normalised [0,1]
    gt_traj: np.ndarray,         # (H, 4): (dx, dy, sin_dth, cos_dth)
    goal_xy: np.ndarray,         # (2,): (x_rel, y_rel) in robot frame
    n_beams: int,
    lidar_max_range: float,      # metres — used to convert normalised scan
    lidar_min_range: float = 0.10,  # metres — self-hit noise floor
    pred_trajs: Optional[np.ndarray] = None,  # (N, H, 4) optional
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw one top-down map onto ax (creates a new figure if ax is None)."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Lidar points (un-normalise)
    hit_x, hit_y, miss_x, miss_y = lidar_to_xy(scan=lidar, n_beams=n_beams)
    # Filter self-hit noise below minimum range (normalised threshold)
    min_norm = lidar_min_range / lidar_max_range
    keep = np.hypot(hit_x, hit_y) >= min_norm
    hit_x, hit_y = hit_x[keep], hit_y[keep]
    # Valid hits: solid dots at actual distance
    ax.scatter(hit_x * lidar_max_range, hit_y * lidar_max_range,
               s=4, c="dimgray", alpha=0.8, label="lidar hit", zorder=1)
    # Max-range / invalid beams: faint dots at sensor boundary (unit circle × max_range)
    ax.scatter(miss_x * lidar_max_range, miss_y * lidar_max_range,
               s=1, c="lightsteelblue", alpha=0.25, label="no return", zorder=1)

    # Predicted trajectories (faint, behind GT)
    if pred_trajs is not None:
        n_pred = pred_trajs.shape[0]
        cmap = cm.get_cmap("Reds")
        for i, pt in enumerate(pred_trajs):
            alpha = 0.35 + 0.45 * (i / max(n_pred - 1, 1))
            color = cmap(0.5 + 0.4 * (i / max(n_pred - 1, 1)))
            xs = np.concatenate([[0.0], pt[:, 0]])
            ys = np.concatenate([[0.0], pt[:, 1]])
            ax.plot(xs, ys, "-", color=color, alpha=alpha, linewidth=1.2, zorder=2)
            ax.scatter(pt[:, 0], pt[:, 1], s=8, color=color, alpha=alpha, zorder=2)
        # Dummy handle for legend
        ax.plot([], [], "-", color=cmap(0.7), alpha=0.7, linewidth=1.5, label=f"pred (N={n_pred})")

    # GT trajectory
    gt_xs = np.concatenate([[0.0], gt_traj[:, 0]])
    gt_ys = np.concatenate([[0.0], gt_traj[:, 1]])
    ax.plot(gt_xs, gt_ys, "b-", linewidth=2.0, label="GT", zorder=3)
    ax.scatter(gt_traj[:, 0], gt_traj[:, 1], s=18, c="blue", zorder=4)

    # Robot origin
    ax.scatter([0], [0], s=80, c="black", marker="^", zorder=5, label="robot")

    # Goal
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=120, c="lime", marker="*",
               zorder=5, label="goal", edgecolors="darkgreen", linewidths=0.8)

    # Robot heading arrow (points along +x = forward)
    ax.annotate("", xy=(0.4, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, markerscale=1.2)
    if title:
        ax.set_title(title, fontsize=9)

    return ax


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(npz_path: str):
    """Return dataset dict with inputs, outputs, meta."""
    data = np.load(npz_path, allow_pickle=True)
    meta = {k: data[k] for k in data.files if k not in ("inputs", "outputs")}
    session_idx   = data["session_idx"]   if "session_idx"   in data.files else None
    session_names = data["session_names"] if "session_names" in data.files else None
    return data["inputs"], data["outputs"], meta, session_idx, session_names


def sample_label(idx: int, session_idx, session_names) -> str:
    """Return a human-readable label for a sample: session name + offset within session."""
    if session_idx is None or session_names is None:
        return f"idx={idx}"
    s = int(session_idx[idx])
    name = str(session_names[s])
    # count offset within this session
    offset = int(np.sum(session_idx[:idx] == s))
    return f"{name}\nsample {offset}"


def parse_meta(meta: dict):
    horizon  = int(meta.get("horizon", 20))
    n_beams  = int(meta.get("n_beams", 360))
    history_len = int(meta.get("history_len", 10))
    include_heading_diff = bool(int(meta.get("include_heading_diff", 0)))
    normalize_positions  = bool(int(meta.get("normalize_positions", 0)))
    workspace_scale      = float(meta.get("workspace_scale", 1.0))
    # non-lidar layout: 2 (curr: v,w) + H*6 (state_hist) + H*2 (cmd_hist) + 2 (goal)
    #                   + 2 (heading_diff, optional)
    non_lidar_dim = (2 + history_len * 6 + history_len * 2 + 2
                     + (2 if include_heading_diff else 0))
    action_dim = int(meta.get("action_dim", 4))
    lidar_max_range_meta = float(meta.get("lidar_max_range", 3.5))
    return horizon, n_beams, non_lidar_dim, normalize_positions, workspace_scale, action_dim, lidar_max_range_meta


def extract_sample(inp: np.ndarray, out: np.ndarray, meta: dict):
    """Split one flat input into (lidar, goal_xy, gt_traj)."""
    horizon, n_beams, non_lidar_dim, normalize_positions, workspace_scale, action_dim, _ = parse_meta(meta)

    non_lidar = inp[:non_lidar_dim]
    lidar     = inp[non_lidar_dim:]    # (n_beams,) normalised [0,1]

    history_len = int(meta.get("history_len", 10))
    action_dim = int(meta.get("action_dim", 4))
    goal_offset = 2 + history_len * 6 + history_len * 2  # start of goal in non_lidar
    goal_xy = non_lidar[goal_offset: goal_offset + 2].copy()

    gt_traj = out.reshape(horizon, action_dim)[:, :4].copy()

    # Un-normalise positions if needed
    if normalize_positions and workspace_scale != 1.0:
        goal_xy        *= workspace_scale
        gt_traj[:, :2] *= workspace_scale

    return lidar, goal_xy, gt_traj, n_beams


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str):
    """Load a diffusion or flow-matching model from a .pth checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = ckpt.get("config", {}).get("model_type", "")

    if "flow_matching" in model_type:
        from flow_matching_model import FlowMatchingConfig, FlowMatchingPolicyModel
        cfg_dict = ckpt["config"]
        cfg = FlowMatchingConfig(
            horizon=cfg_dict["horizon"],
            non_lidar_dim=cfg_dict["non_lidar_dim"],
            n_beams=cfg_dict["n_beams"],
            history_len=cfg_dict.get("history_len", 10),
            include_heading_diff=cfg_dict.get("include_heading_diff", False),
            action_dim=cfg_dict.get("action_dim") or (int(cfg_dict["target_dim"]) // int(cfg_dict["horizon"])),
            lidar_emb_dim=cfg_dict.get("lidar_emb_dim", 64),
            global_cond_dim=cfg_dict.get("global_cond_dim", 256),
            diffusion_step_embed_dim=cfg_dict.get("diffusion_step_embed_dim", 128),
            down_dims=tuple(cfg_dict.get("down_dims", [256, 512, 1024])),
            kernel_size=cfg_dict.get("kernel_size", 3),
            n_groups=cfg_dict.get("n_groups", 8),
            cond_predict_scale=cfg_dict.get("cond_predict_scale", False),
            num_inference_steps=cfg_dict.get("num_inference_steps", 100),
            integrator=cfg_dict.get("integrator", "euler"),
            sigma_min=cfg_dict.get("sigma_min", 1e-4),
        )
        model = FlowMatchingPolicyModel(cfg).to(device)
    else:
        from diffusion_policy_model import DiffusionPolicyConfig, DiffusionPolicyModel
        cfg_dict = ckpt["config"]
        cfg = DiffusionPolicyConfig(
            horizon=cfg_dict["horizon"],
            non_lidar_dim=cfg_dict["non_lidar_dim"],
            n_beams=cfg_dict["n_beams"],
            history_len=cfg_dict.get("history_len", 10),
            include_heading_diff=cfg_dict.get("include_heading_diff", False),
            action_dim=cfg_dict.get("action_dim") or (int(cfg_dict["target_dim"]) // int(cfg_dict["horizon"])),
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

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {model_type or 'model'} from {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise GT and predicted trajectories on a top-down lidar map.")
    parser.add_argument("--dataset",       type=str, required=True,
                        help="Path to .npz dataset (train/val/test)")
    parser.add_argument("--out-dir",       type=str, default="vis_trajectories",
                        help="Output directory for PNG files")
    parser.add_argument("--n-samples",     type=int, default=16,
                        help="Number of random samples to visualise")
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--lidar-max-range", type=float, default=3.5,
                        help="Lidar max range in metres (for un-normalising scans)")
    parser.add_argument("--lidar-min-range", type=float, default=0.10,
                        help="Minimum valid range in metres — filters self-hit noise below this distance")

    # Optional model prediction overlay
    parser.add_argument("--checkpoint",      type=str, default=None,
                        help="Path to .pth checkpoint for model prediction overlay")
    parser.add_argument("--n-pred-samples",  type=int, default=8,
                        help="Number of predicted trajectories to draw per scene")
    parser.add_argument("--num-inference-steps", type=int, default=20,
                        help="Inference steps for model sampling (lower = faster)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Grid layout
    parser.add_argument("--cols", type=int, default=4,
                        help="Columns in the output grid figure")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    inputs, outputs, meta, session_idx, session_names = load_dataset(args.dataset)
    N = inputs.shape[0]
    has_sessions = session_idx is not None
    print(f"Dataset: {args.dataset}  ({N} samples)"
          + (f"  {len(session_names)} sessions" if has_sessions else "  (no session info — re-run create_dataset to add)"))

    indices = np.random.choice(N, size=min(args.n_samples, N), replace=False)
    print(f"Visualising {len(indices)} samples → {args.out_dir}/")

    # Load model if requested
    model = None
    if args.checkpoint:
        model = load_model(args.checkpoint, args.device)

    # --- Parse dataset meta for layout ---
    horizon, n_beams, non_lidar_dim, normalize_positions, workspace_scale, action_dim, lidar_max_range_meta = parse_meta(meta)
    args.lidar_max_range = lidar_max_range_meta  # prefer dataset metadata over CLI default

    # --- Grid figure ---
    n_cols = args.cols
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for plot_i, idx in enumerate(indices):
        ax = axes_flat[plot_i]
        inp = inputs[idx]
        out = outputs[idx]

        lidar, goal_xy, gt_traj, nb = extract_sample(inp, out, meta)

        # Model predictions
        pred_trajs = None
        if model is not None:
            nl_t = torch.from_numpy(inp[:non_lidar_dim]).float().unsqueeze(0).to(args.device)
            li_t = torch.from_numpy(inp[non_lidar_dim:]).float().unsqueeze(0).to(args.device)
            with torch.no_grad():
                samples = [
                    model.sample_trajectory(nl_t, li_t,
                                            num_inference_steps=args.num_inference_steps)
                    .squeeze(0).cpu().numpy()
                    for _ in range(args.n_pred_samples)
                ]
            pred_trajs = np.stack(samples)[:, :, :4]   # (N, H, 4): pos+heading channels
            if normalize_positions and workspace_scale != 1.0:
                pred_trajs[:, :, :2] *= workspace_scale

        render_sample(
            lidar=lidar,
            gt_traj=gt_traj,
            goal_xy=goal_xy,
            n_beams=nb,
            lidar_max_range=args.lidar_max_range,
            lidar_min_range=args.lidar_min_range,
            pred_trajs=pred_trajs,
            title=sample_label(idx, session_idx, session_names),
            ax=ax,
        )

    # Hide unused axes
    for ax in axes_flat[len(indices):]:
        ax.set_visible(False)

    model_tag = Path(args.checkpoint).stem if args.checkpoint else "gt_only"
    fig.suptitle(
        f"{Path(args.dataset).stem}  |  {model_tag}  |  seed={args.seed}",
        fontsize=11,
    )
    fig.tight_layout()

    grid_path = os.path.join(args.out_dir, f"grid_{model_tag}_n{len(indices)}.png")
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"Saved grid: {grid_path}")

    # Also save individual PNGs for closer inspection
    ind_dir = os.path.join(args.out_dir, "individual")
    os.makedirs(ind_dir, exist_ok=True)
    for idx in indices:
        inp = inputs[idx]
        out = outputs[idx]
        lidar, goal_xy, gt_traj, nb = extract_sample(inp, out, meta)

        pred_trajs = None
        if model is not None:
            nl_t = torch.from_numpy(inp[:non_lidar_dim]).float().unsqueeze(0).to(args.device)
            li_t = torch.from_numpy(inp[non_lidar_dim:]).float().unsqueeze(0).to(args.device)
            with torch.no_grad():
                samples = [
                    model.sample_trajectory(nl_t, li_t,
                                            num_inference_steps=args.num_inference_steps)
                    .squeeze(0).cpu().numpy()
                    for _ in range(args.n_pred_samples)
                ]
            pred_trajs = np.stack(samples)
            if normalize_positions and workspace_scale != 1.0:
                pred_trajs[:, :, :2] *= workspace_scale

        fig_i, ax_i = plt.subplots(figsize=(6, 6))
        render_sample(lidar, gt_traj, goal_xy, nb, args.lidar_max_range,
                      lidar_min_range=args.lidar_min_range,
                      pred_trajs=pred_trajs,
                      title=sample_label(idx, session_idx, session_names),
                      ax=ax_i)
        # Use session name in filename when available
        if has_sessions:
            s = int(session_idx[idx])
            sess_name = str(session_names[s])
            offset = int(np.sum(session_idx[:idx] == s))
            fname = f"{sess_name}_s{offset:04d}.png"
        else:
            fname = f"sample_{idx:06d}.png"
        out_path = os.path.join(ind_dir, fname)
        fig_i.tight_layout()
        fig_i.savefig(out_path, dpi=120)
        plt.close(fig_i)

    print(f"Saved {len(indices)} individual PNGs to {ind_dir}/")


if __name__ == "__main__":
    main()
