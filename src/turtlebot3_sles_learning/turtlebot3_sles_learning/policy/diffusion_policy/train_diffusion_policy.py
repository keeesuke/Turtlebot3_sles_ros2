#!/usr/bin/env python3
"""
Train a Diffusion Policy model for local trajectory prediction.

Follows the same dataset interface as train_deterministic_local_traj.py,
but uses DDPM-based diffusion for trajectory generation.

Training:
  - Loss: MSE on predicted noise (epsilon prediction, DDPM)
  - Optimiser: AdamW + ReduceLROnPlateau

Evaluation (per epoch):
  - Training loss (single forward pass, fast)
  - Validation loss + ADE/FDE via full denoising (reduced inference steps)

Inference uses the full DDPM reverse process (100 steps by default).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy_model import (
    DiffusionPolicyConfig,
    DiffusionPolicyModel,
    LocalTrajDataset,
    evaluate_epoch,
    traj_metrics,
)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def wrap_angle(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def lidar_local_to_world(scan: np.ndarray, current_pose: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, th0 = current_pose
    n = scan.shape[0]
    angles_local = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_local = scan * np.cos(angles_local)
    y_local = scan * np.sin(angles_local)
    c, s = np.cos(th0), np.sin(th0)
    x_world = x0 + x_local * c - y_local * s
    y_world = y0 + x_local * s + y_local * c
    return x_world, y_world

def local_deltas_to_world_xy(
    dx_l: np.ndarray, dy_l: np.ndarray, x0: float, y0: float, th0: float
) -> Tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(th0), np.sin(th0)
    return x0 + dx_l * c - dy_l * s, y0 + dx_l * s + dy_l * c

@torch.no_grad()
def save_epoch_sample_plot(
    model: DiffusionPolicyModel,
    dataset: LocalTrajDataset,
    idx: int,
    device: str,
    save_path: str,
    num_inference_steps: int = 100,
) -> None:
    """Qualitative comparison plot: diffusion sample vs expert trajectory."""
    x_inp = dataset.inputs[idx]
    non_lidar = x_inp[: dataset.non_lidar_dim]
    lidar = x_inp[dataset.non_lidar_dim :]

    x0, y0, theta_t = 0.0, 0.0, 0.0  # inputs are local frame; robot at origin, heading right

    horizon = int(dataset.horizon)
    action_dim = int(dataset.action_dim)
    dt = float(dataset.meta.get("dt", 0.1))
    lidar_max_range = float(dataset.meta.get("lidar_max_range", 3.5))
    normalize_lidar = bool(int(dataset.meta.get("normalize_lidar", 1)))  # True for old datasets

    expert_local = dataset.outputs[idx].reshape(horizon, action_dim)[:, :4].copy()

    non_lidar_t = torch.from_numpy(non_lidar).float().unsqueeze(0).to(device)
    lidar_t = torch.from_numpy(lidar).float().unsqueeze(0).to(device)
    model.eval()
    pred_local = (
        model.sample_trajectory(non_lidar_t, lidar_t, num_inference_steps=num_inference_steps)
        .squeeze(0).cpu().numpy()[:, :4]
    )  # (H, 4): position + heading channels

    pred_xw, pred_yw = local_deltas_to_world_xy(pred_local[:, 0], pred_local[:, 1], x0, y0, theta_t)
    exp_xw, exp_yw = local_deltas_to_world_xy(expert_local[:, 0], expert_local[:, 1], x0, y0, theta_t)
    # lidar is metres when normalize_lidar=False, else [0,1] → need to rescale
    lidar_metres = lidar if not normalize_lidar else lidar * lidar_max_range
    lidar_xw, lidar_yw = lidar_local_to_world(lidar_metres, (x0, y0, theta_t))

    dtheta_exp = np.arctan2(expert_local[:, 2], expert_local[:, 3])
    dtheta_pred = np.arctan2(pred_local[:, 2], pred_local[:, 3])
    exp_thw = np.array([wrap_angle(theta_t + d) for d in dtheta_exp])
    pred_thw = np.array([wrap_angle(theta_t + d) for d in dtheta_pred])
    t_axis = dt * (np.arange(horizon) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    axes[0].plot(t_axis, expert_local[:, 0], label="expert dx")
    axes[0].plot(t_axis, pred_local[:, 0], "--", label="pred dx")
    axes[0].set_title("Local dx"); axes[0].grid(True); axes[0].legend()

    axes[1].plot(t_axis, expert_local[:, 1], label="expert dy")
    axes[1].plot(t_axis, pred_local[:, 1], "--", label="pred dy")
    axes[1].set_title("Local dy"); axes[1].grid(True); axes[1].legend()

    axes[2].plot(t_axis, dtheta_exp, label="expert dtheta")
    axes[2].plot(t_axis, dtheta_pred, "--", label="pred dtheta")
    axes[2].set_title("Local dtheta"); axes[2].grid(True); axes[2].legend()

    axes[3].plot(t_axis, exp_thw, label="expert theta")
    axes[3].plot(t_axis, pred_thw, "--", label="pred theta")
    axes[3].set_title("World theta"); axes[3].grid(True); axes[3].legend()

    axes[4].plot(t_axis, expert_local[:, 2], label="expert sin(dtheta)")
    axes[4].plot(t_axis, pred_local[:, 2], "--", label="pred sin(dtheta)")
    axes[4].plot(t_axis, expert_local[:, 3], label="expert cos(dtheta)")
    axes[4].plot(t_axis, pred_local[:, 3], "--", label="pred cos(dtheta)")
    axes[4].set_title("Heading representation"); axes[4].grid(True); axes[4].legend()

    axes[5].plot(exp_xw, exp_yw, "b-", label="expert xy")
    axes[5].plot(pred_xw, pred_yw, "r--", label="pred xy")
    axes[5].scatter([x0], [y0], c="k", s=35, label="current pose")
    axes[5].scatter(lidar_xw, lidar_yw, s=3, c="gray", alpha=0.45, label="lidar")
    axes[5].set_title("World XY trajectory")
    axes[5].set_aspect("equal", adjustable="box")
    axes[5].grid(True); axes[5].legend(loc="upper left")

    fig.suptitle(f"Diffusion Policy | idx={idx} | device={device}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def save_checkpoint(path: str, model: DiffusionPolicyModel, cfg: Dict, dataset_meta: Dict, best_metrics: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "dataset_meta": dataset_meta,
            "best_metrics": best_metrics,
        },
        path,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Diffusion Policy for local trajectory prediction.")

    parser.add_argument(
        "--train-npz",
        type=str,
        default="train_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz",
    )
    parser.add_argument(
        "--val-npz",
        type=str,
        default="val_local_imitation_dataset_dt0.1_h20_hist10_noposnorm_noheaddiff.npz",
    )

    parser.add_argument("--save-dir", type=str, default="models_diffusion_policy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)

    # Model architecture
    parser.add_argument("--lidar-emb-dim", type=int, default=64)
    parser.add_argument("--global-cond-dim", type=int, default=256)
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    parser.add_argument(
        "--down-dims", type=int, nargs="+", default=[256, 512, 1024],
        help="Channel dims for U-Net encoder levels",
    )
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--cond-predict-scale", action="store_true",
                        help="Use FiLM scale+bias (instead of bias-only) conditioning")

    # Diffusion schedule
    parser.add_argument("--num-train-timesteps", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=100,
                        help="Steps for full inference (sampling)")
    parser.add_argument("--num-eval-inference-steps", type=int, default=10,
                        help="Reduced inference steps used during validation (for speed)")
    parser.add_argument("--num-eval-samples", type=int, default=8,
                        help="Number of trajectory samples drawn per scene during validation (best-of-N)")
    parser.add_argument("--val-every", type=int, default=10,
                        help="Run validation every N epochs (0 = never)")
    parser.add_argument("--val-max-batches", type=int, default=20,
                        help="Cap val batches per epoch for speed (0 = use all)")
    parser.add_argument("--full-eval-every", type=int, default=10,
                        help="Run full val set with all samples every N epochs (0 = never)")
    parser.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2",
                        choices=["linear", "squaredcos_cap_v2"])
    parser.add_argument("--prediction-type", type=str, default="epsilon",
                        choices=["epsilon", "sample"])

    # Qualitative plot
    parser.add_argument("--plot-every", type=int, default=5,
                        help="Save a qualitative comparison plot every K epochs (0=off)")

    # Wandb

    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device

    train_ds = LocalTrajDataset(args.train_npz)
    val_ds = LocalTrajDataset(args.val_npz)

    if train_ds.n_beams != val_ds.n_beams or train_ds.non_lidar_dim != val_ds.non_lidar_dim:
        raise ValueError("Train/val dataset input schemas differ.")
    if train_ds.horizon != val_ds.horizon:
        raise ValueError("Train/val horizon differ.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    horizon = int(train_ds.horizon)
    action_dim = int(train_ds.action_dim)

    history_len = int(train_ds.meta.get("history_len", 10))
    include_heading_diff = bool(int(train_ds.meta.get("include_heading_diff", 0)))

    cfg = DiffusionPolicyConfig(
        horizon=horizon,
        non_lidar_dim=train_ds.non_lidar_dim,
        n_beams=train_ds.n_beams,
        history_len=history_len,
        include_heading_diff=include_heading_diff,
        action_dim=action_dim,
        lidar_emb_dim=args.lidar_emb_dim,
        global_cond_dim=args.global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        cond_predict_scale=args.cond_predict_scale,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )

    model = DiffusionPolicyModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    os.makedirs(args.save_dir, exist_ok=True)
    best = {"val_min_ade": float("inf")}
    best_ckpt = os.path.join(args.save_dir, "best_model_diffusion_policy.pth")
    last_ckpt = os.path.join(args.save_dir, "last_model_diffusion_policy.pth")

    cfg_dict = {
        **vars(args),
        "horizon": horizon,
        "target_dim": int(train_ds.target_dim),
        "n_beams": int(train_ds.n_beams),
        "non_lidar_dim": int(train_ds.non_lidar_dim),
        "model_type": "diffusion_policy_unet",
    }

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "ade": [], "fde": [], "min_ade": [], "min_fde": [],
        "diversity": [], "heading_err": [],
    }

    for ep in range(args.epochs):
        print(f"\nEpoch {ep+1}/{args.epochs}")
        model.train()

        total_loss = 0.0
        n_batches = 0

        for non_lidar, lidar, target in tqdm(train_loader, desc="Training", leave=False, dynamic_ncols=True, file=sys.stdout):
            non_lidar = non_lidar.to(device)
            lidar = lidar.to(device)
            target = target.to(device)

            losses = model.compute_loss(non_lidar, lidar, target)
            loss = losses["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            batch_loss = float(loss.item())
            total_loss += batch_loss
            n_batches += 1
            if n_batches % 50 == 0:
                running = total_loss / n_batches
                print(f"  batch {n_batches}/{len(train_loader)}  loss={running:.6f}", flush=True)

        tr_loss = total_loss / max(n_batches, 1)

        print(f"  train loss={tr_loss:.6f}", flush=True)

        history["train_loss"].append(float(tr_loss))

        do_val = args.val_every > 0 and (ep + 1) % args.val_every == 0
        if do_val:
            is_full_eval = args.full_eval_every > 0 and (ep + 1) % args.full_eval_every == 0
            val_max = 0 if is_full_eval else args.val_max_batches
            val_stats = evaluate_epoch(
                model, val_loader, device=device, horizon=horizon,
                num_inference_steps=args.num_eval_inference_steps,
                num_samples=args.num_eval_samples,
                max_batches=val_max,
                action_dim=action_dim,
            )
            eval_tag = "full" if is_full_eval else f"{val_max}b"

            sched.step(val_stats["loss"])

            print(
                f"  val[{eval_tag}] loss={val_stats['loss']:.6f}\n"
                f"  ADE={val_stats['ade']:.4f}  FDE={val_stats['fde']:.4f}\n"
                f"  minADE-{args.num_eval_samples}={val_stats['min_ade']:.4f}"
                f"  minFDE-{args.num_eval_samples}={val_stats['min_fde']:.4f}\n"
                f"  diversity={val_stats['diversity']:.4f}"
                f"  heading_err={val_stats['heading_err']:.4f} rad",
                flush=True,
            )

            history["val_loss"].append(float(val_stats["loss"]))
            for k in ("ade", "fde", "min_ade", "min_fde", "diversity", "heading_err"):
                history[k].append(float(val_stats[k]))

            if val_stats["min_ade"] < best["val_min_ade"]:
                best = {
                    "val_min_ade": float(val_stats["min_ade"]),
                    "val_min_fde": float(val_stats["min_fde"]),
                    "val_ade": float(val_stats["ade"]),
                    "val_loss": float(val_stats["loss"]),
                    "diversity": float(val_stats["diversity"]),
                }
                save_checkpoint(best_ckpt, model, cfg=cfg_dict, dataset_meta=train_ds.meta, best_metrics=best)
                print("  -> saved best checkpoint", flush=True)

        save_checkpoint(last_ckpt, model, cfg=cfg_dict, dataset_meta=train_ds.meta, best_metrics=best)

        # Qualitative sample plot
        plot_img_path = None
        if args.plot_every > 0 and (ep + 1) % args.plot_every == 0:
            plot_idx = int(np.random.randint(0, len(val_ds)))
            plot_img_path = os.path.join(args.save_dir, f"epoch{ep+1:03d}_val_sample{plot_idx}_comparison.png")
            try:
                save_epoch_sample_plot(
                    model=model, dataset=val_ds, idx=plot_idx,
                    device=device, save_path=plot_img_path,
                    num_inference_steps=args.num_inference_steps,
                )
                print(f"  saved epoch plot: {plot_img_path}")
            except Exception as e:
                print(f"  warning: failed to save epoch plot: {e}")
                plot_img_path = None

    # Training curves
    train_epochs = np.arange(1, len(history["train_loss"]) + 1)
    val_epochs = np.arange(args.val_every, args.val_every * len(history["val_loss"]) + 1, args.val_every)

    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, history["train_loss"], label="train loss")
    if history["val_loss"]:
        plt.plot(val_epochs, history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend()
    loss_png = os.path.join(args.save_dir, "training_curves_diffusion_policy_loss.png")
    plt.tight_layout(); plt.savefig(loss_png); plt.close()
    print(f"Saved loss curves: {loss_png}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(val_epochs, history["ade"], label="ADE")
    axes[0, 0].plot(val_epochs, history["min_ade"], label=f"minADE-{args.num_eval_samples}")
    axes[0, 0].set_title("ADE / minADE"); axes[0, 0].grid(True); axes[0, 0].legend()
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("metres")

    axes[0, 1].plot(val_epochs, history["fde"], label="FDE")
    axes[0, 1].plot(val_epochs, history["min_fde"], label=f"minFDE-{args.num_eval_samples}")
    axes[0, 1].set_title("FDE / minFDE"); axes[0, 1].grid(True); axes[0, 1].legend()
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("metres")

    axes[1, 0].plot(val_epochs, history["diversity"])
    axes[1, 0].set_title("Sample Diversity (pairwise L2)"); axes[1, 0].grid(True)
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("metres")

    axes[1, 1].plot(val_epochs, history["heading_err"])
    axes[1, 1].set_title("Heading Error (best sample, rad)"); axes[1, 1].grid(True)
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("radians")

    fig.suptitle("Diffusion Policy — Validation Metrics", fontsize=13)
    fig.tight_layout()
    metrics_png = os.path.join(args.save_dir, "training_curves_diffusion_policy_metrics.png")
    fig.savefig(metrics_png); plt.close(fig)
    print(f"Saved metric curves: {metrics_png}")

    print("\nDone.")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best metrics: {best}")

if __name__ == "__main__":
    main()