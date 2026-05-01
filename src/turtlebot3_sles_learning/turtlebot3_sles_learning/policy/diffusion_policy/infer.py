#!/usr/bin/env python3
"""
Inference script for trained Diffusion Policy, Flow Matching, or Deterministic models.

Loads a checkpoint, runs sampling on a validation (or any) .npz dataset, prints
aggregate metrics (ADE / FDE / minADE / minFDE / diversity), and optionally saves
per-sample trajectory plots.

Usage examples
--------------
# Diffusion policy (best checkpoint)
python infer.py --ckpt models/diffusion_policy/best_model_diffusion_policy.pth \
                --npz dataset/val_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz

# Flow matching
python infer.py --ckpt models/flow_matching/best_model_flow_matching.pth \
                --npz dataset/val_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz

# Deterministic baseline
python infer.py --ckpt models/deterministic/best_model_deterministic_local_traj.pth \
                --npz dataset/val_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz

# Save plots for the first 16 samples
python infer.py --ckpt models/diffusion_policy/best_model_diffusion_policy.pth \
                --npz  dataset/val_*.npz \
                --plot-dir out/infer_plots \
                --plot-samples 16 \
                --num-samples 8 \
                --num-inference-steps 10
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_model_type(ckpt_path: str) -> str:
    """Guess model type from checkpoint filename."""
    name = Path(ckpt_path).stem.lower()
    if "flow" in name:
        return "flow"
    if "deterministic" in name:
        return "deterministic"
    return "diffusion"


def load_model(ckpt_path: str, device: str, model_type: Optional[str] = None):
    """Load a model from a checkpoint. Returns (model, config_dict, dataset_meta)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    meta = ckpt.get("dataset_meta", {})
    best = ckpt.get("best_metrics", {})

    if model_type is None:
        model_type = detect_model_type(ckpt_path)

    if model_type == "diffusion":
        from diffusion_policy_model import DiffusionPolicyConfig, DiffusionPolicyModel
        cfg = DiffusionPolicyConfig(
            horizon=cfg_dict["horizon"],
            non_lidar_dim=cfg_dict["non_lidar_dim"],
            n_beams=cfg_dict["n_beams"],
            history_len=cfg_dict.get("history_len", 10),
            include_heading_diff=cfg_dict.get("include_heading_diff", False),
            action_dim=cfg_dict.get("action_dim", 4),
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
            history_len=cfg_dict.get("history_len", 10),
            include_heading_diff=cfg_dict.get("include_heading_diff", False),
            action_dim=cfg_dict.get("action_dim", 4),
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
            dropout=cfg_dict.get("dropout", 0.1),
        )
        model = DeterministicLocalTrajModel(cfg).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Use diffusion / flow / deterministic.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded {model_type} model from {ckpt_path}")
    if best:
        print(f"  checkpoint best metrics: {best}")

    return model, model_type, cfg_dict, meta


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    model_type: str,
    loader: DataLoader,
    device: str,
    horizon: int,
    num_inference_steps: int,
    num_samples: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    """Run inference and return aggregate metrics."""
    from diffusion_policy_model import traj_metrics

    accum = {"ade": 0.0, "fde": 0.0, "min_ade": 0.0, "min_fde": 0.0,
             "diversity": 0.0, "heading_err": 0.0}
    n_batches = 0

    for non_lidar, lidar, target in loader:
        if max_batches > 0 and n_batches >= max_batches:
            break
        non_lidar = non_lidar.to(device)
        lidar     = lidar.to(device)
        target    = target.to(device)
        B = non_lidar.shape[0]

        if model_type == "deterministic":
            # Single prediction — treat as num_samples=1
            pred = model(non_lidar, lidar).reshape(B, -1)  # (B, H*4)
            samples = pred.unsqueeze(1)  # (B, 1, H*4)
        else:
            samples = torch.stack(
                [model.sample_trajectory(non_lidar, lidar, num_inference_steps=num_inference_steps)
                 .reshape(B, -1)
                 for _ in range(num_samples)],
                dim=1,
            )  # (B, num_samples, H*4)

        m = traj_metrics(samples, target, horizon=horizon, best_of_n=(model_type != "deterministic"))
        for k in ("ade", "fde", "min_ade", "min_fde", "diversity"):
            accum[k] += float(m[k])

        # Heading error on best (or only) sample
        ade_per = torch.norm(
            samples.view(B, -1, horizon, 4)[..., :2]
            - target.view(B, 1, horizon, 4)[..., :2],
            dim=-1,
        ).mean(dim=-1)  # (B, S)
        best_idx = ade_per.argmin(dim=1)
        best_samples = samples[torch.arange(B, device=device), best_idx]
        hm = traj_metrics(best_samples, target, horizon=horizon, best_of_n=False)
        accum["heading_err"] += float(hm["heading_err"])

        n_batches += 1
        if n_batches % 10 == 0:
            print(f"  {n_batches} batches done...", flush=True)

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_sample_plots(
    model,
    model_type: str,
    dataset,
    indices,
    device: str,
    horizon: int,
    num_inference_steps: int,
    num_samples: int,
    plot_dir: str,
):
    """Save trajectory plots for the given dataset indices."""
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    for idx in indices:
        non_lidar, lidar, target = dataset[idx]
        non_lidar = non_lidar.unsqueeze(0).to(device)
        lidar     = lidar.unsqueeze(0).to(device)
        target    = target.to(device)

        with torch.no_grad():
            if model_type == "deterministic":
                preds = [model(non_lidar, lidar).reshape(horizon, 4).cpu().numpy()]
            else:
                preds = [
                    model.sample_trajectory(non_lidar, lidar, num_inference_steps=num_inference_steps)
                    .reshape(horizon, 4).cpu().numpy()
                    for _ in range(num_samples)
                ]

        gt = target.cpu().numpy().reshape(horizon, 4)

        fig, ax = plt.subplots(figsize=(5, 5))
        # Lidar point cloud
        lidar_np = lidar.squeeze(0).cpu().numpy()
        angles = np.linspace(0, 2 * np.pi, len(lidar_np), endpoint=False)
        valid = lidar_np >= 0
        r = lidar_np[valid] * 10.0  # rough scale — adjust to your max range
        lx = r * np.sin(angles[valid])
        ly = r * np.cos(angles[valid])
        ax.scatter(lx, ly, s=1, c="gray", alpha=0.4, label="lidar")

        # Predicted trajectories
        for i, pred in enumerate(preds):
            cum_x = np.cumsum(pred[:, 0])
            cum_y = np.cumsum(pred[:, 1])
            ax.plot(cum_y, cum_x, color="steelblue", alpha=0.5,
                    label="pred" if i == 0 else None, linewidth=1)

        # Ground truth
        gt_cx = np.cumsum(gt[:, 0])
        gt_cy = np.cumsum(gt[:, 1])
        ax.plot(gt_cy, gt_cx, color="red", linewidth=2, label="GT")

        ax.set_aspect("equal")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        out = os.path.join(plot_dir, f"sample_{idx:05d}.png")
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(indices)} plots to {plot_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference for diffusion / flow / deterministic policy models.")
    parser.add_argument("--ckpt",   required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--npz",    required=True, help="Path to dataset .npz file to evaluate on")
    parser.add_argument("--model-type", default=None,
                        help="Model type: diffusion | flow | deterministic (auto-detected from filename if omitted)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=10,
                        help="Denoising / ODE steps at inference (default 10 for speed)")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of trajectories sampled per observation (generative models)")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Cap evaluation at N batches (0 = full dataset)")
    parser.add_argument("--plot-dir", default=None,
                        help="If set, save per-sample trajectory plots here")
    parser.add_argument("--plot-samples", type=int, default=16,
                        help="Number of random samples to plot (used with --plot-dir)")
    args = parser.parse_args()

    model, model_type, cfg_dict, meta = load_model(args.ckpt, args.device, args.model_type)

    from diffusion_policy_model import LocalTrajDataset
    dataset = LocalTrajDataset(args.npz)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, drop_last=False)
    horizon = int(dataset.horizon)

    print(f"\nRunning inference on {args.npz}")
    print(f"  model_type={model_type}  horizon={horizon}  samples={args.num_samples}  "
          f"steps={args.num_inference_steps}  device={args.device}\n")

    metrics = run_inference(
        model=model,
        model_type=model_type,
        loader=loader,
        device=args.device,
        horizon=horizon,
        num_inference_steps=args.num_inference_steps,
        num_samples=args.num_samples,
        max_batches=args.max_batches,
    )

    print("\n--- Results ---")
    print(f"  ADE            : {metrics['ade']:.4f} m")
    print(f"  FDE            : {metrics['fde']:.4f} m")
    print(f"  minADE-{args.num_samples:<2d}       : {metrics['min_ade']:.4f} m")
    print(f"  minFDE-{args.num_samples:<2d}       : {metrics['min_fde']:.4f} m")
    print(f"  Diversity      : {metrics['diversity']:.4f} m")
    print(f"  Heading err    : {metrics['heading_err']:.4f} rad  "
          f"({np.degrees(metrics['heading_err']):.2f} deg)")

    if args.plot_dir:
        n = min(args.plot_samples, len(dataset))
        indices = np.random.choice(len(dataset), size=n, replace=False).tolist()
        save_sample_plots(
            model=model,
            model_type=model_type,
            dataset=dataset,
            indices=indices,
            device=args.device,
            horizon=horizon,
            num_inference_steps=args.num_inference_steps,
            num_samples=args.num_samples,
            plot_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()
