#!/usr/bin/env python3
"""
Data augmentation for real-world NN training sessions.

Reads each session_*_REAL/ folder that does NOT already have a
corresponding session_*_REAL_AUGMENTED/ folder and produces augmented
copies of the raw data.

Augmentation: mirror + goal perturbation (combined).
  - Mirror: flip left/right (LiDAR reversed, ω/y/θ sign-flipped)
  - Goal perturbation: shift final position (= post-hoc goal) by ±10 cm

Output: for each source session, creates 1 augmented sub-session:
  session_*_REAL_AUGMENTED/
    session_<ts>_mirror_goalshift/    ← mirrored + goal shifted

Re-running is safe: sessions with existing _AUGMENTED folders are skipped.
Only newly collected sessions are processed.

Usage:
    python3 augment_training_data.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# Fixed parameters
GOAL_PERTURB_M = 0.10   # ±10 cm uniform shift on goal position
SEED = 123


def mirror_session(positions, orientations, lin_vels, ang_vels,
                   ctrl_lin, ctrl_ang, state_ts, ctrl_ts, lidar_ts,
                   lidar_scans, extra_fields):
    """Return mirrored copies of all arrays (flip across forward axis)."""
    m_positions    = positions.copy()
    m_positions[:, 1] *= -1.0

    m_orientations = -orientations.copy()
    m_lin_vels = lin_vels.copy()
    m_ang_vels = -ang_vels.copy()
    m_ctrl_lin = ctrl_lin.copy()
    m_ctrl_ang = -ctrl_ang.copy()
    m_lidar_scans = [s[::-1].copy() for s in lidar_scans]

    m_extra = {}
    for k, v in extra_fields.items():
        if k == 'target_position':
            tp = v.copy()
            if len(tp) >= 2:
                tp[1] *= -1.0
            m_extra[k] = tp
        else:
            m_extra[k] = v.copy() if hasattr(v, 'copy') else v

    return (m_positions, m_orientations, m_lin_vels, m_ang_vels,
            m_ctrl_lin, m_ctrl_ang, state_ts.copy(), ctrl_ts.copy(),
            lidar_ts.copy(), m_lidar_scans, m_extra)


def perturb_goal(positions, rng):
    """Shift the last position (= post-hoc goal) by a random ±10 cm offset."""
    p = positions.copy()
    p[-1, 0] += rng.uniform(-GOAL_PERTURB_M, GOAL_PERTURB_M)
    p[-1, 1] += rng.uniform(-GOAL_PERTURB_M, GOAL_PERTURB_M)
    return p


def load_session(session_folder, ts):
    """Load raw data from a session folder."""
    npz_path = os.path.join(session_folder, f'robot_data_{ts}.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'Not found: {npz_path}')
    d = np.load(npz_path, allow_pickle=True)

    lidar_path = os.path.join(session_folder, f'lidar_ranges_{ts}.npz')
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f'Not found: {lidar_path}')
    lr = np.load(lidar_path, allow_pickle=True)
    n_scans = int(lr['num_scans'])
    lidar_scans = [lr[f'scan_{i}'] for i in range(n_scans)]

    primary_keys = {'positions', 'orientations', 'linear_velocities',
                    'angular_velocities', 'control_linear', 'control_angular',
                    'robot_state_timestamps', 'control_input_timestamps',
                    'lidar_timestamps'}
    extra = {}
    for k in d.keys():
        if k not in primary_keys:
            extra[k] = d[k]

    return (d['positions'], d['orientations'],
            d['linear_velocities'], d['angular_velocities'],
            d['control_linear'], d['control_angular'],
            d['robot_state_timestamps'], d['control_input_timestamps'],
            d['lidar_timestamps'], lidar_scans, extra)


def save_session(out_folder, ts,
                 positions, orientations, lin_vels, ang_vels,
                 ctrl_lin, ctrl_ang, state_ts, ctrl_ts, lidar_ts,
                 lidar_scans, extra_fields, aug_method):
    """Save augmented session in the same NPZ schema as the recorder."""
    os.makedirs(out_folder, exist_ok=True)

    npz_kwargs = dict(
        positions=positions,
        orientations=orientations,
        linear_velocities=lin_vels,
        angular_velocities=ang_vels,
        control_linear=ctrl_lin,
        control_angular=ctrl_ang,
        robot_state_timestamps=state_ts,
        control_input_timestamps=ctrl_ts,
        lidar_timestamps=lidar_ts,
    )
    for k, v in extra_fields.items():
        npz_kwargs[k] = v
    np.savez(os.path.join(out_folder, f'robot_data_{ts}.npz'), **npz_kwargs)

    lidar_kwargs = {'num_scans': np.array(len(lidar_scans))}
    for i, s in enumerate(lidar_scans):
        lidar_kwargs[f'scan_{i}'] = s
    np.savez(os.path.join(out_folder, f'lidar_ranges_{ts}.npz'), **lidar_kwargs)

    info = {
        'session_type': 'REAL',
        'timestamp': ts,
        'total_robot_states': len(positions),
        'total_control_inputs': len(ctrl_lin),
        'total_lidar_scans': len(lidar_scans),
        'start_position': positions[0].tolist() if len(positions) > 0 else [0, 0],
        'end_position': positions[-1].tolist() if len(positions) > 0 else [0, 0],
        'augmentation': aug_method,
    }
    with open(os.path.join(out_folder, f'session_info_{ts}.json'), 'w') as f:
        json.dump(info, f, indent=2)


def extract_timestamp(folder_name):
    """Extract YYYYMMDD_HHMMSS from session_YYYYMMDD_HHMMSS_REAL."""
    name = os.path.basename(os.path.normpath(folder_name))
    name = name.removeprefix('session_')
    return name.split('_REAL')[0]


def main():
    data_dir = os.path.expanduser('~/robot_data')
    rng = np.random.default_rng(SEED)

    all_dirs = sorted(Path(data_dir).glob('session_*_REAL'))
    originals = [d for d in all_dirs
                 if d.name.endswith('_REAL') and '_AUGMENTED' not in d.name]

    if not originals:
        print('No session_*_REAL folders found in', data_dir)
        sys.exit(1)

    to_process = []
    for d in originals:
        aug_dir = Path(str(d) + '_AUGMENTED')
        if aug_dir.exists():
            continue
        to_process.append(d)

    print(f'Found {len(originals)} original sessions, {len(to_process)} need augmentation.')
    if not to_process:
        print('All sessions already augmented. Nothing to do.')
        return

    total_new = 0

    for session_dir in to_process:
        ts = extract_timestamp(str(session_dir))
        aug_root = Path(str(session_dir) + '_AUGMENTED')
        print(f'\n{"="*60}')
        print(f'Session: {session_dir.name}  (ts={ts})')

        try:
            orig_data = load_session(str(session_dir), ts)
        except (FileNotFoundError, KeyError) as e:
            print(f'  ✗ Skipping: {e}')
            continue

        (positions, orientations, lin_vels, ang_vels,
         ctrl_lin, ctrl_ang, state_ts, ctrl_ts, lidar_ts,
         lidar_scans, extra) = orig_data

        # Mirror + goal shift (combined)
        (m_pos, m_ori, m_lv, m_av, m_cl, m_ca,
         m_sts, m_cts, m_lts, m_lidar, m_extra) = mirror_session(
            positions, orientations, lin_vels, ang_vels,
            ctrl_lin, ctrl_ang, state_ts, ctrl_ts, lidar_ts,
            lidar_scans, extra)
        mgp_pos = perturb_goal(m_pos, rng)
        save_session(str(aug_root / f'session_{ts}_mirror_goalshift'), ts,
                     mgp_pos, m_ori, m_lv, m_av, m_cl, m_ca,
                     m_sts, m_cts, m_lts, m_lidar, m_extra,
                     aug_method='mirror+goalshift')
        print(f'  ✓ mirror+goalshift ({len(positions)} frames)')

        total_new += 1
        print(f'  → 1 augmented sub-session in {aug_root.name}/')

    print(f'\n{"="*60}')
    print(f'Done. Created {total_new} augmented sub-sessions '
          f'from {len(to_process)} sessions.')


if __name__ == '__main__':
    main()
