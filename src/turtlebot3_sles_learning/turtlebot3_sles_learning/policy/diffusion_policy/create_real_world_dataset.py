#!/usr/bin/env python3
"""
Build a diffusion-policy-compatible dataset from real-world TurtleBot3 sessions.

Output format matches LocalTrajDataset (same as create_dataset_imitation_local_traj.py),
but the action representation is:
    target = (dx_1, dy_1, sin_dθ_1, cos_dθ_1, v_1, w_1,   …   dx_H, …, w_H)
           = flattened (H, 6) array of future state waypoints
           where dx/dy are in the robot frame at time t,
           v/w are normalised state velocities (v/v_max, w/w_max).

This matches the MPC output format: diffusion policy runs at 10 Hz and outputs
a waypoint sequence that the Kanayama controller (50 Hz) tracks directly.
action_dim=6, horizon=20 → 2.0 s lookahead at 10 Hz.

Input layout (non_lidar vector) matches StructuredConditionEncoder:
    [0:6]                   current state  (x,y, sinθ, cosθ, v/v_max, w/w_max)
    [6 : 6+H_hist*6]        state history  H_hist × (x,y, sinθ, cosθ, v/v_max, w/w_max)
    [6+H_hist*6 : ..+H_hist*2]  cmd history H_hist × (v/v_max, w/w_max)
    [..+2]                  goal relative  (x_g_rel, y_g_rel) in robot frame
    last n_beams elements   lidar scan in metres [0, lidar_max_range] (raw, not normalised by default)

Usage:
    python create_real_world_dataset.py \\
        --data-root ~/robot_data \\
        --out-dir   ~/robot_data/real_world_datasets/diffusion_run01 \\
        --horizon 20 --history-len 10 --dt 0.1 --lidar-max-range 3.5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (match training data)
# ---------------------------------------------------------------------------
DEFAULT_V_MAX        = 0.26    # m/s
DEFAULT_W_MAX        = 1.82    # rad/s
DEFAULT_DT           = 0.10    # seconds per waypoint (10 Hz, matches MPC frequency)
DEFAULT_HORIZON      = 20      # future waypoints to predict (20 × 0.1 s = 2.0 s)
DEFAULT_HISTORY_LEN  = 10      # past state/cmd steps to include
DEFAULT_LIDAR_MAX    = 3.5     # metres (raw lidar max range)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _world_to_robot_xy(dx_w: float, dy_w: float, theta: float) -> Tuple[float, float]:
    c, s = np.cos(theta), np.sin(theta)
    return dx_w * c + dy_w * s, -dx_w * s + dy_w * c


def _process_lidar(scan: np.ndarray, max_range: float, normalize: bool = False) -> np.ndarray:
    """
    Sanitize and optionally normalize a lidar scan.

    normalize=False (default): clip to [0, max_range], bad returns → max_range.
                               Values are in metres — same units as goal/position.
    normalize=True:            divide by max_range → [0, 1].  Legacy behaviour.
    """
    out = np.asarray(scan, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = max_range
    np.clip(out, 0.0, max_range, out=out)
    if normalize:
        out /= max_range
    # Resample to exactly 360 beams if needed
    n = len(out)
    if n != 360:
        idx = np.linspace(0, n - 1, 360)
        out = np.interp(idx, np.arange(n), out).astype(np.float32)
    return out


def _nearest_before(timestamps: np.ndarray, query: float) -> int:
    """Largest index i s.t. timestamps[i] <= query.  Returns 0 if none."""
    idx = np.searchsorted(timestamps, query, side='right') - 1
    return max(int(idx), 0)


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def load_session(session_folder: Path) -> Optional[Dict]:
    ts_candidates = [
        p.stem.replace('robot_data_', '')
        for p in session_folder.glob('robot_data_*.npz')
    ]
    if not ts_candidates:
        return None
    ts = ts_candidates[0]

    robot_path = session_folder / f'robot_data_{ts}.npz'
    lidar_path = session_folder / f'lidar_ranges_{ts}.npz'
    if not robot_path.exists() or not lidar_path.exists():
        return None

    r  = np.load(robot_path, allow_pickle=True)
    lr = np.load(lidar_path, allow_pickle=True)

    n_scans = int(lr['num_scans'])
    scans = [lr[f'scan_{i}'] for i in range(n_scans)]

    return {
        'positions':            np.asarray(r['positions'],            dtype=np.float64),
        'orientations':         np.asarray(r['orientations'],         dtype=np.float64),
        'linear_velocities':    np.asarray(r['linear_velocities'],    dtype=np.float64),
        'angular_velocities':   np.asarray(r['angular_velocities'],   dtype=np.float64),
        'control_linear':       np.asarray(r['control_linear'],       dtype=np.float64),
        'control_angular':      np.asarray(r['control_angular'],      dtype=np.float64),
        'robot_state_timestamps':   np.asarray(r['robot_state_timestamps'], dtype=np.float64),
        'control_input_timestamps': np.asarray(r['control_input_timestamps'], dtype=np.float64),
        'lidar_timestamps':     np.asarray(r['lidar_timestamps'],     dtype=np.float64),
        'lidar_scans':          scans,
        # Goal: use target_position if non-zero, else last position
        'target_position':      np.asarray(r.get('target_position', r['positions'][-1][:2]),
                                           dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Sample builder
# ---------------------------------------------------------------------------

def build_samples(
    session: Dict,
    horizon: int,
    history_len: int,
    dt: float,
    v_max: float,
    w_max: float,
    lidar_max_range: float,
    normalize_lidar: bool = False,
    min_speed: float = 0.01,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (inputs, outputs) arrays for one session.

    Anchors are placed on a uniform dt grid regardless of the raw recording
    rate.  All data lookups use ZOH (nearest-preceding timestamp), so this
    works correctly whether dt < or > the recording rate (e.g. 50 Hz dataset
    from 20 Hz raw data, or 10 Hz dataset from 20 Hz raw data).
    """
    pos      = session['positions']
    ori      = session['orientations']
    lin_vel  = session['linear_velocities']
    ang_vel  = session['angular_velocities']
    ctrl_lin = session['control_linear']
    ctrl_ang = session['control_angular']
    state_ts = session['robot_state_timestamps']
    ctrl_ts  = session['control_input_timestamps']
    lidar_ts = session['lidar_timestamps']
    lidar_raw = session['lidar_scans']

    tgt = session['target_position']
    goal_x_w = float(tgt[0]) if tgt[0] != 0 else float(pos[-1, 0])
    goal_y_w = float(tgt[1]) if len(tgt) > 1 and tgt[1] != 0 else float(pos[-1, 1])

    lidar_norm = [_process_lidar(s, lidar_max_range, normalize=normalize_lidar) for s in lidar_raw]

    n_beams       = 360
    curr_dim      = 2  # (v, w) — pose is identity in local frame
    state_dim     = 6
    cmd_dim       = 2
    non_lidar_dim = curr_dim + history_len * state_dim + history_len * cmd_dim + 2
    input_dim     = non_lidar_dim + n_beams
    output_dim    = horizon * 6  # (dx, dy, sin_dθ, cos_dθ, v, w) per step

    inputs_list:  List[np.ndarray] = []
    outputs_list: List[np.ndarray] = []

    # Uniform anchor grid at the desired dt — independent of recording rate.
    # ZOH handles upsampling (50 Hz from 20 Hz) and downsampling equally.
    t_data_start = max(state_ts[0], ctrl_ts[0])
    t_data_end   = min(state_ts[-1], ctrl_ts[-1])
    t_start = t_data_start + history_len * dt
    t_end   = t_data_end   - horizon * dt
    if t_end <= t_start:
        return np.zeros((0, input_dim), np.float32), np.zeros((0, output_dim), np.float32)

    for t in np.arange(t_start, t_end, dt):
        # Current state (ZOH from odometry)
        s_idx = _nearest_before(state_ts, t)

        # Current lidar (nearest scan within 300 ms)
        l_idx = int(np.argmin(np.abs(lidar_ts - t)))
        if abs(lidar_ts[l_idx] - t) > 0.30:
            continue

        x_t     = float(pos[s_idx, 0])
        y_t     = float(pos[s_idx, 1])
        theta_t = float(ori[s_idx])
        v_t     = float(lin_vel[s_idx])
        w_t     = float(ang_vel[s_idx])

        # Motion filter: use ZOH command at t
        ci = _nearest_before(ctrl_ts, t)
        if abs(ctrl_lin[ci]) < min_speed and abs(ctrl_ang[ci]) < min_speed * 6:
            continue

        sin_t = np.sin(theta_t)
        cos_t = np.cos(theta_t)
        g_rel_x, g_rel_y = _world_to_robot_xy(goal_x_w - x_t, goal_y_w - y_t, theta_t)
        curr = np.array([v_t / v_max, w_t / w_max], dtype=np.float32)

        c0, s0 = float(np.cos(theta_t)), float(np.sin(theta_t))

        # State history: H steps spaced dt apart looking back (ZOH), in local robot frame
        shist = []
        for j in range(history_len):
            k  = history_len - j
            sk = _nearest_before(state_ts, t - k * dt)
            x_k = float(pos[sk, 0])
            y_k = float(pos[sk, 1])
            theta_k = float(ori[sk])
            dx_w = x_k - x_t
            dy_w = y_k - y_t
            dx_l =  dx_w * c0 + dy_w * s0
            dy_l = -dx_w * s0 + dy_w * c0
            dtheta_k = (theta_k - theta_t + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793
            shist.extend([
                dx_l, dy_l,
                float(np.sin(dtheta_k)), float(np.cos(dtheta_k)),
                float(lin_vel[sk]) / v_max, float(ang_vel[sk]) / w_max,
            ])

        # Command history: H steps spaced dt apart looking back (ZOH)
        chist = []
        for j in range(history_len):
            k  = history_len - j
            ck = _nearest_before(ctrl_ts, t - k * dt)
            chist.extend([float(ctrl_lin[ck]) / v_max, float(ctrl_ang[ck]) / w_max])

        inp = np.concatenate([
            curr,
            np.array(shist,  dtype=np.float32),
            np.array(chist,  dtype=np.float32),
            np.array([g_rel_x, g_rel_y], dtype=np.float32),
            lidar_norm[l_idx],
        ]).astype(np.float32)

        # Future waypoints: H steps spaced dt apart looking forward.
        # Each waypoint = (dx, dy, sin_dθ, cos_dθ, v_norm, w_norm) in robot frame at t.
        future = []
        for k in range(1, horizon + 1):
            fk = _nearest_before(state_ts, t + k * dt)
            x_k     = float(pos[fk, 0])
            y_k     = float(pos[fk, 1])
            theta_k = float(ori[fk])
            v_k     = float(lin_vel[fk])
            w_k     = float(ang_vel[fk])
            dx_w = x_k - x_t
            dy_w = y_k - y_t
            dx_l =  dx_w * c0 + dy_w * s0
            dy_l = -dx_w * s0 + dy_w * c0
            dtheta = (theta_k - theta_t + np.pi) % (2 * np.pi) - np.pi
            future.extend([
                dx_l, dy_l,
                float(np.sin(dtheta)), float(np.cos(dtheta)),
                v_k / v_max, w_k / w_max,
            ])

        out = np.array(future, dtype=np.float32)
        assert out.shape[0] == output_dim

        inputs_list.append(inp)
        outputs_list.append(out)

        if max_samples is not None and len(inputs_list) >= max_samples:
            break

    if not inputs_list:
        return np.zeros((0, input_dim), np.float32), np.zeros((0, output_dim), np.float32)

    return (np.stack(inputs_list).astype(np.float32),
            np.stack(outputs_list).astype(np.float32))


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    session_folders: List[Path],
    horizon: int,
    history_len: int,
    dt: float,
    v_max: float,
    w_max: float,
    lidar_max_range: float,
    normalize_lidar: bool = False,
    max_samples_per_session: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_inputs, all_outputs, all_sidx, session_names = [], [], [], []

    for folder in tqdm(session_folders, desc="Sessions"):
        sess = load_session(folder)
        if sess is None:
            continue
        inp, out = build_samples(
            sess, horizon, history_len, dt, v_max, w_max, lidar_max_range,
            normalize_lidar=normalize_lidar,
            max_samples=max_samples_per_session,
        )
        if inp.shape[0] == 0:
            continue
        idx = len(session_names)
        session_names.append(folder.name)
        all_inputs.append(inp)
        all_outputs.append(out)
        all_sidx.append(np.full(inp.shape[0], idx, dtype=np.int32))

    if not all_inputs:
        raise RuntimeError("No valid samples found in any session folder.")

    return (np.concatenate(all_inputs,  axis=0),
            np.concatenate(all_outputs, axis=0),
            np.concatenate(all_sidx,    axis=0),
            np.array(session_names, dtype=object))


# ---------------------------------------------------------------------------
# Save in LocalTrajDataset format
# ---------------------------------------------------------------------------

def save_npz(
    path: Path,
    inputs: np.ndarray,
    outputs: np.ndarray,
    session_idx: np.ndarray,
    session_names: np.ndarray,
    horizon: int,
    history_len: int,
    dt: float,
    v_max: float,
    w_max: float,
    lidar_max_range: float,
    normalize_lidar: bool = False,
    n_beams: int = 360,
) -> None:
    non_lidar_dim = inputs.shape[1] - n_beams
    np.savez_compressed(
        path,
        inputs=inputs,
        outputs=outputs,
        session_idx=session_idx,
        session_names=session_names,
        # Metadata read by LocalTrajDataset and PolicyRunner
        horizon=np.int32(horizon),
        n_beams=np.int32(n_beams),
        input_dim=np.int32(inputs.shape[1]),
        output_dim=np.int32(outputs.shape[1]),
        history_len=np.int32(history_len),
        dt=np.float32(dt),
        v_max=np.float32(v_max),
        w_max=np.float32(w_max),
        lidar_max_range=np.float32(lidar_max_range),
        normalize_lidar=np.int32(1 if normalize_lidar else 0),
        action_dim=np.int32(6),
        include_heading_diff=np.int32(0),
        normalize_positions=np.int32(0),
    )
    kb = os.path.getsize(path) // 1024
    print(f"  {path.name}: {inputs.shape[0]} samples, {kb} KB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build diffusion-policy dataset (action_dim=2) from real TurtleBot3 sessions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-root', default='~/robot_data',
                   help='Root dir containing session_*_REAL (and _AUGMENTED) folders.')
    p.add_argument('--out-dir',   default='~/robot_data/real_world_datasets/diffusion',
                   help='Output directory for train/val/test npz files.')
    p.add_argument('--horizon',     type=int,   default=DEFAULT_HORIZON)
    p.add_argument('--history-len', type=int,   default=DEFAULT_HISTORY_LEN)
    p.add_argument('--dt',          type=float, default=DEFAULT_DT)
    p.add_argument('--v-max',       type=float, default=DEFAULT_V_MAX)
    p.add_argument('--w-max',       type=float, default=DEFAULT_W_MAX)
    p.add_argument('--lidar-max-range', type=float, default=DEFAULT_LIDAR_MAX)
    p.add_argument('--train-frac',  type=float, default=0.80)
    p.add_argument('--val-frac',    type=float, default=0.10)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--include-augmented', action='store_true',
                   help='Also include session_*_REAL_AUGMENTED sub-sessions.')
    p.add_argument('--normalize-lidar', action='store_true',
                   help='Normalize lidar to [0,1] by lidar_max_range (legacy). '
                        'Default: store raw metres so lidar and goal are in the same units.')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    data_root = Path(os.path.expanduser(args.data_root))
    out_dir   = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect session folders
    sessions: List[Path] = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith('_REAL') and '_AUGMENTED' not in name:
            sessions.append(d)
        elif args.include_augmented and name.endswith('_REAL_AUGMENTED'):
            sessions.extend(sorted(d.glob('session_*')))

    if not sessions:
        print(f"No session_*_REAL folders found in {data_root}", file=sys.stderr)
        sys.exit(1)

    # Shuffle and split by session (not by sample) to avoid leakage
    rng = np.random.default_rng(args.seed)
    sessions = [sessions[i] for i in rng.permutation(len(sessions))]
    n = len(sessions)
    n_train = max(1, int(n * args.train_frac))
    n_val   = max(1, int(n * args.val_frac))
    splits = {
        'train': sessions[:n_train],
        'val':   sessions[n_train:n_train + n_val],
        'test':  sessions[n_train + n_val:],
    }

    print(f"Sessions: {n} total  train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}")
    print(f"Horizon={args.horizon} steps ({args.horizon * args.dt:.1f}s)  "
          f"history_len={args.history_len}  dt={args.dt}s")
    print(f"v_max={args.v_max} w_max={args.w_max} lidar_max={args.lidar_max_range}m  "
          f"normalize_lidar={args.normalize_lidar}")

    max_per = 200 if args.dry_run else None

    for split_name, split_sessions in splits.items():
        if not split_sessions:
            print(f"  {split_name}: no sessions — skipped")
            continue
        print(f"\nBuilding {split_name} ({len(split_sessions)} sessions)…")
        inputs, outputs, sidx, snames = build_dataset(
            split_sessions,
            horizon=args.horizon,
            history_len=args.history_len,
            dt=args.dt,
            v_max=args.v_max,
            w_max=args.w_max,
            lidar_max_range=args.lidar_max_range,
            normalize_lidar=args.normalize_lidar,
            max_samples_per_session=max_per,
        )
        tag = f"_h{args.horizon}_hist{args.history_len}"
        save_npz(
            out_dir / f"{split_name}_dataset{tag}.npz",
            inputs, outputs, sidx, snames,
            horizon=args.horizon,
            history_len=args.history_len,
            dt=args.dt,
            v_max=args.v_max,
            w_max=args.w_max,
            lidar_max_range=args.lidar_max_range,
            normalize_lidar=args.normalize_lidar,
        )

    print("\nDone.")
    print(f"Dataset saved to: {out_dir}")
    print(f"\nTo train:")
    print(f"  cd <diffusion_policy_dir>")
    tag = f"_h{args.horizon}_hist{args.history_len}"
    print(f"  python train_diffusion_policy.py \\")
    print(f"    --train-npz {out_dir}/train_dataset{tag}.npz \\")
    print(f"    --val-npz   {out_dir}/val_dataset{tag}.npz \\")
    print(f"    --save-dir  models/diffusion_real_world \\")
    print(f"    --epochs 100 --batch-size 512 --lr 1e-4 \\")
    print(f"    --wandb-project diffusion-policy")


if __name__ == '__main__':
    main()
