#!/usr/bin/env python3
"""
Train a deterministic baseline model for local trajectory prediction.

Mapping:
  (current state + history + command history + target + lidar) -> future trajectory

Outputs:
  - future trajectory over horizon H:
      (dx, dy, sin(dtheta), cos(dtheta)) in robot frame
  - stored as flattened (H*4) in the dataset, but the model operates on (B, H, 4).

Loss:
  - L_traj  : MSE on trajectory (x0 space)
  - L_smooth: second-difference smoothness on dx, dy
  - Total:   L = L_traj + lambda_smooth * L_smooth

Evaluation:
  - train / val loss
  - ADE / FDE on validation set
  - qualitative plots of predicted vs GT trajectories
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

from deterministic_local_traj import (
    DeterministicConfig,
    DeterministicLocalTrajModel,
    LocalTrajDataset,
    evaluate_epoch,
    trajectory_loss,
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
    c = np.cos(th0)
    s = np.sin(th0)
    x_world = x0 + x_local * c - y_local * s
    y_world = y0 + x_local * s + y_local * c
    return x_world, y_world

def local_deltas_to_world_xy(
    dx_l: np.ndarray, dy_l: np.ndarray, x0: float, y0: float, th0: float
) -> Tuple[np.ndarray, np.ndarray]:
    c = np.cos(th0)
    s = np.sin(th0)
    dx_w = dx_l * c - dy_l * s
    dy_w = dx_l * s + dy_l * c
    return x0 + dx_w, y0 + dy_w

@torch.no_grad()
def save_epoch_sample_plot(
    model: DeterministicLocalTrajModel,
    dataset: LocalTrajDataset,
    idx: int,
    device: str,
    save_path: str,
) -> None:
    """
    Qualitative plot similar to DDPM training, but using the deterministic model.
    """
    x_inp = dataset.inputs[idx]
    non_lidar = x_inp[: dataset.non_lidar_dim]
    lidar = x_inp[dataset.non_lidar_dim :]

    x0, y0, theta_t = 0.0, 0.0, 0.0  # inputs are local frame; robot at origin, heading right

    horizon = int(dataset.horizon)
    action_dim = int(dataset.action_dim)
    dt = float(dataset.meta.get("dt", 0.1))
    lidar_max_range = float(dataset.meta.get("lidar_max_range", 3.5))

    # Expert trajectory
    expert_local = dataset.outputs[idx].reshape(horizon, action_dim)[:, :4].copy()

    # Deterministic prediction
    non_lidar_t = torch.from_numpy(non_lidar).float().unsqueeze(0).to(device)
    lidar_t = torch.from_numpy(lidar).float().unsqueeze(0).to(device)
    model.eval()
    pred_local = model(non_lidar_t, lidar_t).detach().cpu().numpy().reshape(horizon, action_dim)[:, :4]

    # World XY for expert and prediction
    pred_xw, pred_yw = local_deltas_to_world_xy(pred_local[:, 0], pred_local[:, 1], x0, y0, theta_t)
    exp_xw, exp_yw = local_deltas_to_world_xy(expert_local[:, 0], expert_local[:, 1], x0, y0, theta_t)
    lidar_xw, lidar_yw = lidar_local_to_world(lidar * lidar_max_range, (x0, y0, theta_t))

    dtheta_exp = np.arctan2(expert_local[:, 2], expert_local[:, 3])
    dtheta_pred = np.arctan2(pred_local[:, 2], pred_local[:, 3])
    exp_thw = np.array([wrap_angle(theta_t + d) for d in dtheta_exp], dtype=np.float32)
    pred_thw = np.array([wrap_angle(theta_t + d) for d in dtheta_pred], dtype=np.float32)

    t_axis = dt * (np.arange(horizon) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    axes[0].plot(t_axis, expert_local[:, 0], label="expert dx")
    axes[0].plot(t_axis, pred_local[:, 0], "--", label="pred dx")
    axes[0].set_title("Local dx")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_axis, expert_local[:, 1], label="expert dy")
    axes[1].plot(t_axis, pred_local[:, 1], "--", label="pred dy")
    axes[1].set_title("Local dy")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t_axis, dtheta_exp, label="expert dtheta")
    axes[2].plot(t_axis, dtheta_pred, "--", label="pred dtheta")
    axes[2].set_title("Local dtheta")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t_axis, exp_thw, label="expert theta")
    axes[3].plot(t_axis, pred_thw, "--", label="pred theta")
    axes[3].set_title("World theta")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(t_axis, expert_local[:, 2], label="expert sin(dtheta)")
    axes[4].plot(t_axis, pred_local[:, 2], "--", label="pred sin(dtheta)")
    axes[4].plot(t_axis, expert_local[:, 3], label="expert cos(dtheta)")
    axes[4].plot(t_axis, pred_local[:, 3], "--", label="pred cos(dtheta)")
    axes[4].set_title("Heading representation")
    axes[4].grid(True)
    axes[4].legend()

    axes[5].plot(exp_xw, exp_yw, "b-", label="expert xy")
    axes[5].plot(pred_xw, pred_yw, "r--", label="pred xy")
    axes[5].scatter([x0], [y0], c="k", s=35, label="current pose")
    axes[5].scatter(lidar_xw, lidar_yw, s=3, c="gray", alpha=0.45, label="current lidar")
    axes[5].set_title("World XY trajectory")
    axes[5].set_aspect("equal", adjustable="box")
    axes[5].grid(True)
    axes[5].legend(loc="upper left")

    fig.suptitle(f"Deterministic baseline | idx={idx} | device={device}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def save_checkpoint(path: str, model: DeterministicLocalTrajModel, cfg: Dict, dataset_meta: Dict, best_metrics: Dict) -> None:
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
    parser = argparse.ArgumentParser(description="Train deterministic baseline for local trajectory prediction.")

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

    parser.add_argument("--save-dir", type=str, default="models_deterministic_local_traj")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Model dims
    parser.add_argument("--lidar-emb-dim", type=int, default=64)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Loss weights
    parser.add_argument("--lambda-smooth", type=float, default=0.01, help="Weight for second-diff smoothness on dx/dy.")

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    train_losses = []
    val_losses = []
    val_ade_history = []
    val_fde_history = []

    for ep in range(args.epochs):
        print(f"\nEpoch {ep+1}/{args.epochs}")
        model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_smooth = 0.0
        n_batches = 0

        for non_lidar, lidar, target in tqdm(train_loader, desc="Training", leave=False,
                                             dynamic_ncols=True, file=sys.stdout):
            non_lidar = non_lidar.to(device)
            lidar = lidar.to(device)
            target = target.to(device)

            pred_traj = model(non_lidar, lidar)  # (B, H, 4)
            losses = trajectory_loss(pred_traj, target, horizon=horizon, lambda_smooth=args.lambda_smooth)

            loss = losses["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            total_mse += float(losses["traj_mse"].item())
            total_smooth += float(losses["smooth"].item())
            n_batches += 1

            if n_batches % 50 == 0:
                running = total_loss / n_batches
                print(f"  batch {n_batches}/{len(train_loader)}  loss={running:.6f}", flush=True)

        denom = max(n_batches, 1)
        tr_loss = total_loss / denom
        tr_mse = total_mse / denom
        tr_smooth = total_smooth / denom

        val_stats = evaluate_epoch(model, val_loader, device=device, horizon=horizon)
        va_loss = val_stats["loss"]
        va_ade = val_stats["ade"]
        va_fde = val_stats["fde"]

        sched.step(va_loss)

        print(
            f"  train: loss={tr_loss:.6f} traj_mse={tr_mse:.6f} smooth={tr_smooth:.6f}\n"
            f"  val  : loss={va_loss:.6f} traj_mse={val_stats['traj_mse']:.6f} "
            f"smooth={val_stats['smooth']:.6f} ade={va_ade:.6f} fde={va_fde:.6f}"
        )

        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))
        val_ade_history.append(float(va_ade))
        val_fde_history.append(float(va_fde))

        if va_ade < best["val_ade"]:
            best = {
                "val_ade": float(va_ade),
                "val_fde": float(va_fde),
                "val_loss": float(va_loss),
            }
            save_checkpoint(best_ckpt, model, cfg=cfg_dict, dataset_meta=train_ds.meta, best_metrics=best)
            print("  -> saved best checkpoint")

        save_checkpoint(last_ckpt, model, cfg=cfg_dict, dataset_meta=train_ds.meta, best_metrics=best)

        if args.plot_every and args.plot_every > 0 and ((ep + 1) % args.plot_every == 0):
            plot_idx = int(np.random.randint(0, len(val_ds)))
            out_png = os.path.join(args.save_dir, f"epoch{ep+1:03d}_val_sample{plot_idx}_comparison.png")
            try:
                save_epoch_sample_plot(
                    model=model,
                    dataset=val_ds,
                    idx=plot_idx,
                    device=device,
                    save_path=out_png,
                )
                print(f"  saved epoch plot: {out_png}")
            except Exception as e:
                print(f"  warning: failed to save epoch plot: {e}")

    # Plot training curves: loss and ADE/FDE history.
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    curves_path = os.path.join(args.save_dir, "training_curves_deterministic_local_traj_loss.png")
    plt.tight_layout()
    plt.savefig(curves_path)
    plt.close()
    print(f"Saved loss curves: {curves_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_ade_history, label="val ADE")
    plt.plot(epochs, val_fde_history, label="val FDE")
    plt.xlabel("epoch")
    plt.ylabel("distance")
    plt.grid(True)
    plt.legend()
    curves_path_ade = os.path.join(args.save_dir, "training_curves_deterministic_local_traj_ade_fde.png")
    plt.tight_layout()
    plt.savefig(curves_path_ade)
    plt.close()
    print(f"Saved ADE/FDE curves: {curves_path_ade}")

    print("\nDone.")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best metrics: {best}")

if __name__ == "__main__":
    main()
