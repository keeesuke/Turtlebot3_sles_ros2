#!/usr/bin/env python3
"""
Preprocessing pipeline for real-world NN imitation-learning data.

Input:  session folder produced by robot_data_recorder_real_world.py
        ~/robot_data/session_YYYYMMDD_HHMMSS_REAL/
           robot_data_TIMESTAMP.npz        (positions, orientations, velocities)
           lidar_ranges_TIMESTAMP.npz      (scan_0, scan_1, …)

Output: training datasets in  ~/robot_data/real_world_datasets/TIMESTAMP/
           train_dataset.npz
           val_dataset.npz
           test_dataset.npz

Processing steps:
  1. Load raw NPZ files
  2. Post-hoc goal assignment  (last recorded position = goal)
  3. Transform goal to robot frame for every timestep
  4. Sync LiDAR timestamps to state timestamps (nearest-neighbour)
  5. Normalize LiDAR  (inf/nan/≤0 → lidar_max_range; clip; resample to 360)
  6. Filter invalid frames  (near-zero velocity, missing LiDAR)
  7. Build training arrays   [v, ω] | [goal_x_r, goal_y_r] | lidar(360)  →  [v_cmd, ω_cmd]
  8. Shuffle + split  80 / 10 / 10  (train / val / test)

Usage:
  python3 prepare_training_data_real_world.py  SESSION_FOLDER  [OPTIONS]

  python3 prepare_training_data_real_world.py ~/robot_data/session_20260301_120000_REAL
  python3 prepare_training_data_real_world.py ~/robot_data/session_*_REAL --merge
  python3 prepare_training_data_real_world.py ~/robot_data/session_20260301_120000_REAL \\
      --output-dir ~/robot_data/real_world_datasets/run01 \\
      --lidar-max-range 1.0 \\
      --min-speed 0.01 \\
      --lidar-sync-tolerance 0.15 \\
      --split 0.8 0.1 0.1
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_timestamp(session_folder: str) -> str:
    """Extract YYYYMMDD_HHMMSS from folder name like session_YYYYMMDD_HHMMSS_REAL."""
    name = os.path.basename(os.path.normpath(session_folder))
    # Remove leading "session_" and trailing "_REAL" (or "_REAL_*")
    name = name.removeprefix('session_')
    # Split on "_REAL" and take the first part
    ts = name.split('_REAL')[0]
    return ts


def _load_raw_npz(session_folder: str, timestamp: str):
    """Load main NPZ file from session folder."""
    path = os.path.join(session_folder, f'robot_data_{timestamp}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'robot_data NPZ not found: {path}')
    data = np.load(path, allow_pickle=True)
    print(f'  ✓ Loaded robot_data_{timestamp}.npz')
    return data


def _load_lidar_npz(session_folder: str, timestamp: str):
    """Load lidar_ranges NPZ; returns list of arrays (one per scan)."""
    path = os.path.join(session_folder, f'lidar_ranges_{timestamp}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'lidar_ranges NPZ not found: {path}')
    lr = np.load(path, allow_pickle=True)
    n  = int(lr['num_scans'])
    scans = [lr[f'scan_{i}'] for i in range(n)]
    print(f'  ✓ Loaded lidar_ranges_{timestamp}.npz  ({n} scans)')
    return scans


def _goal_to_robot_frame(goal_x_w: float, goal_y_w: float,
                          robot_x: np.ndarray, robot_y: np.ndarray,
                          robot_theta: np.ndarray) -> np.ndarray:
    """
    Transform world-frame goal (scalar) into robot frame for each timestep.

    Returns: (N, 2) array of [goal_x_robot, goal_y_robot].
    """
    dx = goal_x_w - robot_x
    dy = goal_y_w - robot_y
    cos_t = np.cos(-robot_theta)
    sin_t = np.sin(-robot_theta)
    gx_r = dx * cos_t - dy * sin_t
    gy_r = dx * sin_t + dy * cos_t
    return np.stack([gx_r, gy_r], axis=1)


def _normalize_lidar(ranges: np.ndarray, max_range: float) -> np.ndarray:
    """
    Normalize a single LiDAR scan:
      - inf / nan / ≤ 0  →  max_range
      - clip to [0, max_range]
      - resample to exactly 360 rays
    """
    r = np.array(ranges, dtype=np.float32)
    bad = ~np.isfinite(r) | (r <= 0.0)
    r[bad] = max_range
    np.clip(r, 0.0, max_range, out=r)
    n = len(r)
    if n != 360:
        idx = np.linspace(0, n - 1, 360)
        r = np.interp(idx, np.arange(n), r).astype(np.float32)
    return r


def _sync_lidar_to_states(state_ts: np.ndarray,
                           lidar_ts: np.ndarray,
                           lidar_scans: list,
                           tolerance_sec: float) -> tuple:
    """
    For each state timestep, find the nearest LiDAR scan.

    Returns:
        synced_lidar : (N, 360) float32  —  matched scan per state frame
        valid_mask   : (N,) bool         —  False where nearest scan > tolerance
    """
    N = len(state_ts)
    synced = np.zeros((N, 360), dtype=np.float32)
    valid  = np.ones(N, dtype=bool)

    if len(lidar_ts) == 0:
        print('  ⚠ No LiDAR timestamps — all frames marked invalid.')
        return synced, np.zeros(N, dtype=bool)

    for i, ts in enumerate(state_ts):
        idx      = int(np.argmin(np.abs(lidar_ts - ts)))
        dt       = abs(lidar_ts[idx] - ts)
        if dt > tolerance_sec:
            valid[i] = False
        else:
            synced[i] = lidar_scans[idx]

    return synced, valid


def _filter_frames(control_linear: np.ndarray,
                   control_angular: np.ndarray,
                   lidar_valid: np.ndarray,
                   min_speed: float) -> np.ndarray:
    """
    Build a boolean mask of frames worth training on:
      - LiDAR sync is valid
      - Robot was actually moving (avoids degenerate near-stop samples)
    """
    moving = (np.abs(control_linear) > min_speed) | (np.abs(control_angular) > min_speed * 2.0)
    return lidar_valid & moving


def _split_and_save(states:           np.ndarray,
                    target_positions: np.ndarray,
                    lidar_scans:      np.ndarray,
                    control_linear:   np.ndarray,
                    control_angular:  np.ndarray,
                    output_dir:       str,
                    split:            tuple,
                    seed:             int = 42):
    """
    Shuffle and split into train/val/test, then save each as NPZ.

    NPZ keys match what train_mlp.py ImitationLearningDataset expects:
        states           (N, 2)   [v, omega]
        target_positions (N, 2)   goal in robot frame
        lidar_scans      (N, 360)
        control_linear   (N,)
        control_angular  (N,)
    """
    N = len(states)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    tr_frac, val_frac, _ = split
    n_train = int(N * tr_frac)
    n_val   = int(N * val_frac)

    splits = {
        'train': idx[:n_train],
        'val':   idx[n_train:n_train + n_val],
        'test':  idx[n_train + n_val:],
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_idx in splits.items():
        path = os.path.join(output_dir, f'{split_name}_dataset.npz')
        np.savez(path,
                 states=states[split_idx].astype(np.float32),
                 target_positions=target_positions[split_idx].astype(np.float32),
                 lidar_scans=lidar_scans[split_idx].astype(np.float32),
                 control_linear=control_linear[split_idx].astype(np.float32),
                 control_angular=control_angular[split_idx].astype(np.float32))
        n = len(split_idx)
        size_kb = os.path.getsize(path) // 1024
        print(f'  ✓ {split_name}_dataset.npz  ({n} samples, {size_kb} KB)')

    # Summary JSON
    summary = {
        'total_samples': N,
        'train': int(splits['train'].shape[0]),
        'val':   int(splits['val'].shape[0]),
        'test':  int(splits['test'].shape[0]),
        'input_dim':  364,
        'output_dim': 2,
    }
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  ✓ dataset_info.json')


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def process_session(session_folder: str,
                    lidar_max_range: float,
                    min_speed: float,
                    lidar_sync_tolerance: float) -> dict | None:
    """
    Process one session folder and return numpy arrays for merging.

    Returns None if session has insufficient data.
    """
    session_folder = os.path.abspath(os.path.expanduser(session_folder))
    print(f'\nProcessing: {session_folder}')

    ts = _extract_timestamp(session_folder)
    print(f'  Timestamp: {ts}')

    # Load raw data
    try:
        raw  = _load_raw_npz(session_folder, ts)
        lidar_scans_raw = _load_lidar_npz(session_folder, ts)
    except FileNotFoundError as e:
        print(f'  ✗ {e}')
        return None

    N_states = len(raw['positions'])
    N_lidar  = len(lidar_scans_raw)
    print(f'  State frames : {N_states}')
    print(f'  LiDAR scans  : {N_lidar}')

    if N_states < 10:
        print('  ✗ Too few state frames — skipping session.')
        return None
    if N_lidar < 1:
        print('  ✗ No LiDAR scans — skipping session.')
        return None

    positions    = raw['positions']       # (N, 2)
    orientations = raw['orientations']    # (N,)
    lin_vels     = raw['linear_velocities']   # (N,)
    ang_vels     = raw['angular_velocities']  # (N,)
    ctrl_lin     = raw['control_linear']  # (M,) — may differ from N
    ctrl_ang     = raw['control_angular']
    state_ts     = raw['robot_state_timestamps']    # (N,)
    ctrl_ts      = raw['control_input_timestamps']  # (M,)
    lidar_ts     = raw['lidar_timestamps']          # (K,)

    # ── 1. Normalize LiDAR scans ──────────────────────────────────────────
    print(f'  Normalizing LiDAR  (max_range={lidar_max_range} m) …')
    lidar_normalized = [_normalize_lidar(s, lidar_max_range) for s in lidar_scans_raw]

    # ── 2. Sync LiDAR to state timestamps ────────────────────────────────
    print(f'  Syncing LiDAR to states  (tolerance={lidar_sync_tolerance} s) …')
    synced_lidar, lidar_valid = _sync_lidar_to_states(
        state_ts, lidar_ts, lidar_normalized, lidar_sync_tolerance
    )

    # ── 3. Sync control commands to state timestamps ──────────────────────
    #    (control may be recorded at different rate; match nearest)
    print('  Syncing control commands to states …')
    synced_ctrl_lin = np.zeros(N_states, dtype=np.float32)
    synced_ctrl_ang = np.zeros(N_states, dtype=np.float32)
    ctrl_valid      = np.ones(N_states, dtype=bool)

    if len(ctrl_ts) == 0:
        print('  ⚠ No control timestamps — all control frames marked invalid.')
        ctrl_valid[:] = False
    else:
        for i, ts_i in enumerate(state_ts):
            idx = int(np.argmin(np.abs(ctrl_ts - ts_i)))
            dt  = abs(ctrl_ts[idx] - ts_i)
            if dt > 0.20:   # 200 ms tolerance for controls
                ctrl_valid[i] = False
            else:
                synced_ctrl_lin[i] = ctrl_lin[idx]
                synced_ctrl_ang[i] = ctrl_ang[idx]

    # ── 4. Post-hoc goal assignment ───────────────────────────────────────
    goal_x_w = float(positions[-1, 0])
    goal_y_w = float(positions[-1, 1])
    print(f'  Goal (last frame): ({goal_x_w:.3f}, {goal_y_w:.3f}) m')

    target_positions = _goal_to_robot_frame(
        goal_x_w, goal_y_w, positions[:, 0], positions[:, 1], orientations
    )  # (N, 2)

    # ── 5. Filter valid frames ────────────────────────────────────────────
    combined_valid = lidar_valid & ctrl_valid
    motion_mask = _filter_frames(
        synced_ctrl_lin, synced_ctrl_ang, combined_valid, min_speed
    )
    n_valid = int(motion_mask.sum())
    n_total = len(motion_mask)
    print(f'  Valid frames after filtering: {n_valid} / {n_total} '
          f'({100.0 * n_valid / max(n_total, 1):.1f} %)')

    if n_valid < 50:
        print('  ✗ Too few valid frames after filtering — skipping session.')
        return None

    # ── 6. Assemble training arrays ───────────────────────────────────────
    states_arr = np.stack([lin_vels[motion_mask],
                            ang_vels[motion_mask]], axis=1).astype(np.float32)

    return {
        'states':           states_arr,
        'target_positions': target_positions[motion_mask].astype(np.float32),
        'lidar_scans':      synced_lidar[motion_mask].astype(np.float32),
        'control_linear':   synced_ctrl_lin[motion_mask].astype(np.float32),
        'control_angular':  synced_ctrl_ang[motion_mask].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare real-world training data for NN imitation learning.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'session_folders', nargs='+',
        help='One or more session folders (supports glob patterns).',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for dataset NPZ files. '
             'Default: ~/robot_data/real_world_datasets/TIMESTAMP/',
    )
    parser.add_argument(
        '--lidar-max-range', type=float, default=1.0,
        help='Clip LiDAR readings to this value (m). Must match training-time setting.',
    )
    parser.add_argument(
        '--min-speed', type=float, default=0.01,
        help='Minimum |v| or |ω| to keep frame (filters stopped-robot samples).',
    )
    parser.add_argument(
        '--lidar-sync-tolerance', type=float, default=0.15,
        help='Max allowed time difference (s) between state and LiDAR timestamps.',
    )
    parser.add_argument(
        '--split', nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Train / Val / Test split fractions (must sum to 1.0).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for shuffling.',
    )
    parser.add_argument(
        '--merge', action='store_true',
        help='Merge multiple sessions into a single dataset (default: each session separately).',
    )
    args = parser.parse_args()

    # Validate split
    s = args.split
    if abs(sum(s) - 1.0) > 1e-6:
        print(f'ERROR: Split fractions must sum to 1.0, got {sum(s):.4f}')
        sys.exit(1)

    # Expand globs
    session_folders = []
    for pattern in args.session_folders:
        expanded = sorted(glob.glob(os.path.expanduser(pattern)))
        session_folders.extend(expanded if expanded else [pattern])
    session_folders = [f for f in session_folders if os.path.isdir(f)]
    if not session_folders:
        print('ERROR: No valid session folders found.')
        sys.exit(1)

    print('=' * 60)
    print(f'Real-World Training Data Preparation')
    print(f'Sessions : {len(session_folders)}')
    print(f'LiDAR max range : {args.lidar_max_range} m')
    print(f'Min speed filter: {args.min_speed} m/s (or rad/s)')
    print(f'LiDAR sync tol  : {args.lidar_sync_tolerance} s')
    print(f'Split           : train={s[0]}, val={s[1]}, test={s[2]}')
    print('=' * 60)

    if args.merge:
        # ── Merge all sessions into one dataset ────────────────────────────
        all_data = {k: [] for k in
                    ['states', 'target_positions', 'lidar_scans',
                     'control_linear', 'control_angular']}

        for folder in session_folders:
            result = process_session(
                folder, args.lidar_max_range, args.min_speed, args.lidar_sync_tolerance
            )
            if result is not None:
                for k in all_data:
                    all_data[k].append(result[k])

        if not all_data['states']:
            print('\n✗ No valid data found in any session. Exiting.')
            sys.exit(1)

        merged = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
        total  = len(merged['states'])
        print(f'\nMerged total: {total} samples from {len(session_folders)} sessions')

        if args.output_dir is None:
            from datetime import datetime
            ts_now = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = os.path.join(
                os.path.expanduser('~'), 'robot_data',
                'real_world_datasets', f'merged_{ts_now}'
            )
        else:
            out_dir = os.path.expanduser(args.output_dir)

        print(f'\nSaving to: {out_dir}')
        _split_and_save(
            merged['states'], merged['target_positions'],
            merged['lidar_scans'],
            merged['control_linear'], merged['control_angular'],
            out_dir, tuple(s), args.seed,
        )

    else:
        # ── One dataset per session ────────────────────────────────────────
        for folder in session_folders:
            result = process_session(
                folder, args.lidar_max_range, args.min_speed, args.lidar_sync_tolerance
            )
            if result is None:
                continue

            ts_folder = _extract_timestamp(folder)
            if args.output_dir is None:
                out_dir = os.path.join(
                    os.path.expanduser('~'), 'robot_data',
                    'real_world_datasets', ts_folder
                )
            else:
                out_dir = os.path.expanduser(args.output_dir)

            print(f'\nSaving to: {out_dir}')
            _split_and_save(
                result['states'], result['target_positions'],
                result['lidar_scans'],
                result['control_linear'], result['control_angular'],
                out_dir, tuple(s), args.seed,
            )

    print('\n✅ Done.')


if __name__ == '__main__':
    main()
