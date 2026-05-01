#!/usr/bin/env python3
"""
Generate a dataset for imitation learning of a local trajectory prediction policy.

For each sample at anchor time `t` (from control_input_timestamps):
  Inputs:
    - current pose/state in world frame:
        (x_t, y_t, sin(theta_t), cos(theta_t), v_t, w_t)
    - state history (past `history_len` steps, dt seconds each), ordered oldest->most-recent:
        (x, y, sin(theta), cos(theta), v, w)  [world frame]
    - command history (past `history_len` steps, dt seconds each), ordered oldest->most-recent:
        (v_cmd, w_cmd)
    - target relative goal position in current robot frame:
        (x_g_rel, y_g_rel)
    - optional heading difference in current robot frame:
        (sin(dtheta_goal), cos(dtheta_goal))
    - current lidar scan, assumed normalized to [0,1] (invalid/missing -> 1)

  Outputs:
    - future local trajectory for horizon `horizon` steps at dt seconds each:
        for k=1..horizon: (dx_k, dy_k, sin(dtheta_k), cos(dtheta_k))
      where (dx_k, dy_k) are the SE(2) world->robot-frame displacements relative to pose at time t.

All timestamps are sampled using "latest timestamp <= query" (lt-eps) to align fields
to the same anchor time t.
"""

import argparse
from pathlib import Path
import os

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


DEFAULT_HISTORY_LEN = 10
DEFAULT_HORIZON = 20
DEFAULT_DT = 0.1


def wrap_angle(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def find_largest_timestamp_index(timestamps: np.ndarray, target_timestamp: float) -> int:
    """
    Largest index i such that timestamps[i] <= target_timestamp.
    Returns -1 if no such index exists.
    """
    valid_indices = np.where(timestamps <= target_timestamp)[0]
    if len(valid_indices) == 0:
        return -1
    return int(valid_indices[-1])


def world_xy_to_robot_xy(dx_world: float, dy_world: float, theta0: float) -> Tuple[float, float]:
    """
    SE(2) world->robot-frame rotation by theta0 (pose at time t).
    """
    c = np.cos(theta0)
    s = np.sin(theta0)
    dx_l = dx_world * c + dy_world * s
    dy_l = -dx_world * s + dy_world * c
    return dx_l, dy_l


def sanitize_lidar_scan(scan: np.ndarray) -> np.ndarray:
    """
    Replace invalid/missing returns with max range (=1.0) and clamp to [0,1].
    """
    scan = scan.astype(np.float32, copy=True)
    scan[scan == -1.0] = 1.0
    scan = np.clip(scan, 0.0, 1.0)
    return scan


def load_session_data(session_folder: Path) -> Optional[Dict]:
    """
    Load one session folder.

    Expected keys in robot_data_*.npz:
      - control_linear, control_angular, control_input_timestamps
      - positions (T,2), orientations (T,)
      - linear_velocities, angular_velocities
      - robot_state_timestamps
      - lidar_timestamps
      - target_position (2,)

    Expected contents in lidar_ranges_*.npz:
      - num_scans
      - scan_i arrays for i in [0, num_scans)
    """
    session_folder = Path(session_folder)
    robot_data_files = list(session_folder.glob("robot_data_*.npz"))
    lidar_files = list(session_folder.glob("lidar_ranges_*.npz"))
    if not robot_data_files or not lidar_files:
        return None

    robot_data = np.load(robot_data_files[0], allow_pickle=True)
    lidar_data = np.load(lidar_files[0], allow_pickle=True)

    required_robot = [
        "control_linear",
        "control_angular",
        "control_input_timestamps",
        "positions",
        "orientations",
        "linear_velocities",
        "angular_velocities",
        "robot_state_timestamps",
        "lidar_timestamps",
        "target_position",
    ]
    for k in required_robot:
        if k not in robot_data.files:
            raise KeyError(f"Missing key '{k}' in {robot_data_files[0]}")

    if "num_scans" not in lidar_data.files:
        raise KeyError(f"Missing key 'num_scans' in {lidar_files[0]}")
    num_scans = int(lidar_data["num_scans"])

    scans = []
    for i in range(num_scans):
        scan_key = f"scan_{i}"
        if scan_key not in lidar_data:
            raise KeyError(f"Missing {scan_key} in {lidar_files[0]}")
        scans.append(lidar_data[scan_key])

    lidar_scans = np.asarray(scans, dtype=np.float32)

    return {
        "control_linear": np.asarray(robot_data["control_linear"]),
        "control_angular": np.asarray(robot_data["control_angular"]),
        "control_input_timestamps": np.asarray(robot_data["control_input_timestamps"]),
        "positions": np.asarray(robot_data["positions"], dtype=np.float32),
        "orientations": np.asarray(robot_data["orientations"], dtype=np.float32),
        "linear_velocities": np.asarray(robot_data["linear_velocities"], dtype=np.float32),
        "angular_velocities": np.asarray(robot_data["angular_velocities"], dtype=np.float32),
        "robot_state_timestamps": np.asarray(robot_data["robot_state_timestamps"], dtype=np.float32),
        "lidar_timestamps": np.asarray(robot_data["lidar_timestamps"], dtype=np.float32),
        "lidar_scans": lidar_scans,
        "target_position": np.asarray(robot_data["target_position"], dtype=np.float32),
    }


def create_samples_from_session(
    session_data: Dict,
    history_len: int,
    horizon: int,
    dt: float,
    include_heading_diff: bool,
    normalize_positions: bool,
    v_max: float,
    w_max: float,
    workspace_scale: float,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (inputs, outputs) arrays for one session.
    """
    control_linear = session_data["control_linear"]
    control_angular = session_data["control_angular"]
    control_ts = session_data["control_input_timestamps"]

    positions = session_data["positions"]
    orientations = session_data["orientations"]
    linear_velocities = session_data["linear_velocities"]
    angular_velocities = session_data["angular_velocities"]
    robot_state_ts = session_data["robot_state_timestamps"]

    lidar_ts = session_data["lidar_timestamps"]
    lidar_scans = session_data["lidar_scans"]

    target_position = session_data["target_position"]

    n_beams = int(lidar_scans.shape[1]) if lidar_scans.ndim == 2 else int(lidar_scans.shape[-1])

    x_size = 6  # (x,y,sin,cos,v,w)
    state_hist_dim = history_len * x_size
    cmd_hist_dim = history_len * 2
    curr_dim = 6  # current (x,y,sin,cos,v,w)
    target_dim = 2
    heading_dim = 2 if include_heading_diff else 0
    input_dim = curr_dim + state_hist_dim + cmd_hist_dim + target_dim + heading_dim + n_beams
    output_dim = horizon * 4

    inputs_list = []
    outputs_list = []

    last_state_idx = len(robot_state_ts) - 1
    if last_state_idx < 0:
        return np.zeros((0, input_dim), dtype=np.float32), np.zeros((0, output_dim), dtype=np.float32)

    for i in range(len(control_ts)):
        t = float(control_ts[i])

        # Current state aligned at "latest <= t"
        state_idx = find_largest_timestamp_index(robot_state_ts, t)
        if state_idx == -1:
            continue

        # Current lidar aligned at "latest <= t"
        lidar_idx = find_largest_timestamp_index(lidar_ts, t)
        if lidar_idx == -1:
            continue

        x_t, y_t = float(positions[state_idx][0]), float(positions[state_idx][1])
        theta_t = float(orientations[state_idx])
        v_t = float(linear_velocities[state_idx])
        w_t = float(angular_velocities[state_idx])

        sin_t = float(np.sin(theta_t))
        cos_t = float(np.cos(theta_t))

        # Target relative goal position (in robot frame at time t)
        dx_world = float(target_position[0] - x_t)
        dy_world = float(target_position[1] - y_t)
        x_g_rel, y_g_rel = world_xy_to_robot_xy(dx_world, dy_world, theta_t)

        # Lidar scan
        scan = sanitize_lidar_scan(lidar_scans[lidar_idx])

        # Normalize
        if normalize_positions:
            x_t_n = x_t / float(workspace_scale)
            y_t_n = y_t / float(workspace_scale)
            x_g_rel_n = x_g_rel / float(workspace_scale)
            y_g_rel_n = y_g_rel / float(workspace_scale)
        else:
            x_t_n, y_t_n = x_t, y_t
            x_g_rel_n, y_g_rel_n = x_g_rel, y_g_rel

        v_t_n = v_t / float(v_max)
        w_t_n = w_t / float(w_max)

        # State history (oldest -> most recent), past `history_len` steps excluding t
        state_hist = []
        for j in range(history_len):
            k = history_len - j  # oldest uses k=history_len
            query_time = t - k * dt
            s_idx_k = find_largest_timestamp_index(robot_state_ts, query_time)
            if s_idx_k == -1:
                s_idx_k = 0  # repeat earliest available state

            x_k, y_k = float(positions[s_idx_k][0]), float(positions[s_idx_k][1])
            theta_k = float(orientations[s_idx_k])
            v_k = float(linear_velocities[s_idx_k])
            w_k = float(angular_velocities[s_idx_k])

            if normalize_positions:
                x_k_n = x_k / float(workspace_scale)
                y_k_n = y_k / float(workspace_scale)
            else:
                x_k_n, y_k_n = x_k, y_k

            sin_k = float(np.sin(theta_k))
            cos_k = float(np.cos(theta_k))
            v_k_n = v_k / float(v_max)
            w_k_n = w_k / float(w_max)
            state_hist.extend([x_k_n, y_k_n, sin_k, cos_k, v_k_n, w_k_n])

        # Command history (oldest -> most recent), pad with zeros if missing
        cmd_hist = []
        for j in range(history_len):
            k = history_len - j
            query_time = t - k * dt
            c_idx_k = find_largest_timestamp_index(control_ts, query_time)
            if c_idx_k == -1:
                cmd_hist.extend([0.0, 0.0])
            else:
                vcmd = float(control_linear[c_idx_k])
                wcmd = float(control_angular[c_idx_k])
                cmd_hist.extend([vcmd / float(v_max), wcmd / float(w_max)])

        # Optional heading difference in robot frame: direction-to-goal angle relative to robot forward (+x).
        heading_diff = []
        if include_heading_diff:
            r = float(np.hypot(x_g_rel_n, y_g_rel_n))
            if r < 1e-9:
                heading_diff = [0.0, 1.0]
            else:
                sin_d = y_g_rel_n / r
                cos_d = x_g_rel_n / r
                heading_diff = [float(sin_d), float(cos_d)]

        curr_part = [x_t_n, y_t_n, sin_t, cos_t, v_t_n, w_t_n]
        target_part = [x_g_rel_n, y_g_rel_n]

        inp = np.concatenate(
            [
                np.asarray(curr_part, dtype=np.float32),
                np.asarray(state_hist, dtype=np.float32),
                np.asarray(cmd_hist, dtype=np.float32),
                np.asarray(target_part, dtype=np.float32),
                np.asarray(heading_diff, dtype=np.float32),
                scan.astype(np.float32, copy=False).reshape(-1),
            ],
            axis=0,
        )
        if inp.shape[0] != input_dim:
            raise RuntimeError(f"input_dim mismatch: got {inp.shape[0]} expected {input_dim}")

        # Outputs: future local-frame relative trajectory over horizon steps, k=1..horizon
        future_steps = []
        c0 = float(np.cos(theta_t))
        s0 = float(np.sin(theta_t))
        for k in range(1, horizon + 1):
            query_time = t + k * dt
            f_idx = find_largest_timestamp_index(robot_state_ts, query_time)
            if f_idx == -1:
                f_idx = 0
            elif query_time > robot_state_ts[last_state_idx]:
                f_idx = last_state_idx

            x_k = float(positions[f_idx][0])
            y_k = float(positions[f_idx][1])
            theta_k = float(orientations[f_idx])

            dx_w = x_k - x_t
            dy_w = y_k - y_t
            dx_l = dx_w * c0 + dy_w * s0
            dy_l = -dx_w * s0 + dy_w * c0

            dtheta = wrap_angle(theta_k - theta_t)
            sin_d = float(np.sin(dtheta))
            cos_d = float(np.cos(dtheta))

            if normalize_positions:
                dx_l = dx_l / float(workspace_scale)
                dy_l = dy_l / float(workspace_scale)

            future_steps.extend([dx_l, dy_l, sin_d, cos_d])

        out = np.asarray(future_steps, dtype=np.float32)
        if out.shape[0] != output_dim:
            raise RuntimeError(f"output_dim mismatch: got {out.shape[0]} expected {output_dim}")

        inputs_list.append(inp)
        outputs_list.append(out)

        if max_samples is not None and len(inputs_list) >= max_samples:
            break

    if not inputs_list:
        return np.zeros((0, input_dim), dtype=np.float32), np.zeros((0, output_dim), dtype=np.float32)

    inputs_arr = np.stack(inputs_list).astype(np.float32, copy=False)
    outputs_arr = np.stack(outputs_list).astype(np.float32, copy=False)
    return inputs_arr, outputs_arr


def create_dataset_from_folders(
    session_folders: List[Path],
    split_name: str,
    history_len: int,
    horizon: int,
    dt: float,
    include_heading_diff: bool,
    normalize_positions: bool,
    v_max: float,
    w_max: float,
    workspace_scale: float,
    max_samples_per_session: Optional[int],
):
    all_inputs = []
    all_outputs = []
    all_session_idx = []   # integer index into session_names for each sample
    session_names = []     # ordered list of session folder names that produced samples

    for session_folder in tqdm(session_folders, desc=f"Building {split_name}"):
        session_data = load_session_data(session_folder)
        if session_data is None:
            continue

        inputs, outputs = create_samples_from_session(
            session_data=session_data,
            history_len=history_len,
            horizon=horizon,
            dt=dt,
            include_heading_diff=include_heading_diff,
            normalize_positions=normalize_positions,
            v_max=v_max,
            w_max=w_max,
            workspace_scale=workspace_scale,
            max_samples=max_samples_per_session,
        )
        if inputs.shape[0] == 0:
            continue

        sess_idx = len(session_names)
        session_names.append(session_folder.name)
        all_inputs.append(inputs)
        all_outputs.append(outputs)
        all_session_idx.append(np.full(inputs.shape[0], sess_idx, dtype=np.int32))

    if not all_inputs:
        return None

    inputs_arr = np.concatenate(all_inputs, axis=0)
    outputs_arr = np.concatenate(all_outputs, axis=0)
    session_idx_arr = np.concatenate(all_session_idx, axis=0)
    session_names_arr = np.array(session_names, dtype=object)
    return inputs_arr, outputs_arr, session_idx_arr, session_names_arr


def sanity_check_dataset(inputs: np.ndarray, outputs: np.ndarray, horizon: int) -> None:
    if inputs.ndim != 2:
        raise ValueError(f"inputs should be 2D, got shape {inputs.shape}")
    if outputs.ndim != 2:
        raise ValueError(f"outputs should be 2D, got shape {outputs.shape}")
    if outputs.shape[1] != horizon * 4:
        raise ValueError(f"outputs dim mismatch: got {outputs.shape[1]} expected {horizon * 4}")

    # Check sin^2 + cos^2 for each output step (numerical tolerance).
    outs = outputs.reshape(outputs.shape[0], horizon, 4)
    sin_d = outs[..., 2]
    cos_d = outs[..., 3]
    unit = sin_d * sin_d + cos_d * cos_d
    if not np.all(np.isfinite(unit)):
        raise ValueError("Non-finite values in sin/cos outputs")
    max_err = float(np.max(np.abs(unit - 1.0)))
    if max_err > 1e-3:
        raise ValueError(f"sin^2+cos^2 unit check failed: max_err={max_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate imitation-learning local trajectory dataset.")

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/rant3/robot_data_expert/data_sets",
        help="Root directory containing session_* folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for train/val/test .npz files. Defaults to --data-root.",
    )

    parser.add_argument("--train-folders", type=int, default=700)
    parser.add_argument("--val-folders", type=int, default=77)
    parser.add_argument("--test-folders", type=int, default=77)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--history-len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)

    parser.add_argument("--v-max", type=float, default=0.26, help="Velocity normalization: v / v_max")
    parser.add_argument("--w-max", type=float, default=1.5, help="Angular velocity normalization: w / w_max")
    parser.add_argument(
        "--workspace-scale",
        type=float,
        default=1.0,
        help="Position normalization scale; used only if --normalize-positions is set.",
    )

    parser.add_argument("--normalize-positions", action="store_true", help="Normalize positions/deltas by workspace-scale")
    parser.add_argument("--include-heading-diff", action="store_true", help="Include optional (sin,cos) heading difference to goal")

    parser.add_argument("--dry-run", action="store_true", help="Generate a small subset for quick validation.")
    parser.add_argument("--dry-run-samples", type=int, default=100, help="Max samples per session in --dry-run mode.")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else data_root
    os.makedirs(out_dir, exist_ok=True)

    # Collect session folders (mirrors existing dataset scripts).
    session_folders_all = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("session_")],
        key=lambda x: x.name,
    )
    valid_folders = []
    for folder in session_folders_all:
        if list(folder.glob("robot_data_*.npz")) and list(folder.glob("lidar_ranges_*.npz")):
            valid_folders.append(folder)

    if not valid_folders:
        raise RuntimeError(f"No valid session_* folders found under {data_root}")

    np.random.seed(args.random_seed)
    shuffled_indices = np.random.permutation(len(valid_folders))
    valid_folders = [valid_folders[i] for i in shuffled_indices]

    total_requested = args.train_folders + args.val_folders + args.test_folders
    if total_requested > len(valid_folders):
        # Scale down splits to available folders.
        train_folders = int(len(valid_folders) * args.train_folders / total_requested)
        val_folders = int(len(valid_folders) * args.val_folders / total_requested)
        test_folders = len(valid_folders) - train_folders - val_folders
    else:
        train_folders = args.train_folders
        val_folders = args.val_folders
        test_folders = args.test_folders

    train_list = valid_folders[:train_folders]
    val_list = valid_folders[train_folders : train_folders + val_folders]
    test_list = valid_folders[train_folders + val_folders : train_folders + val_folders + test_folders]

    max_samples_per_session = None
    if args.dry_run:
        max_samples_per_session = int(args.dry_run_samples)

    def dataset_out_path(split: str) -> Path:
        norm_tag = "posnorm" if args.normalize_positions else "noposnorm"
        head_tag = "headdiff" if args.include_heading_diff else "noheaddiff"
        return out_dir / f"{split}_local_imitation_dataset_dt{args.dt:g}_h{args.horizon}_hist{args.history_len}_{norm_tag}_{head_tag}.npz"

    def run_split(session_list: List[Path], split: str):
        res = create_dataset_from_folders(
            session_folders=session_list,
            split_name=split,
            history_len=args.history_len,
            horizon=args.horizon,
            dt=args.dt,
            include_heading_diff=args.include_heading_diff,
            normalize_positions=args.normalize_positions,
            v_max=args.v_max,
            w_max=args.w_max,
            workspace_scale=args.workspace_scale,
            max_samples_per_session=max_samples_per_session,
        )
        if res is None:
            return
        inputs_arr, outputs_arr, session_idx_arr, session_names_arr = res
        print(f"{split}: inputs={inputs_arr.shape} outputs={outputs_arr.shape} "
              f"sessions={len(session_names_arr)}")
        sanity_check_dataset(inputs_arr, outputs_arr, horizon=args.horizon)

        # Infer number of lidar beams from flattened input size and known feature layout.
        curr_dim = 6  # (x,y,sin,cos,v,w)
        state_hist_dim = args.history_len * 6
        cmd_hist_dim = args.history_len * 2
        target_dim = 2
        heading_dim = 2 if args.include_heading_diff else 0
        n_beams = int(inputs_arr.shape[1] - (curr_dim + state_hist_dim + cmd_hist_dim + target_dim + heading_dim))

        np.savez_compressed(
            dataset_out_path(split),
            inputs=inputs_arr,
            outputs=outputs_arr,
            session_idx=session_idx_arr,
            session_names=session_names_arr,
            dt=np.asarray(args.dt, dtype=np.float32),
            history_len=np.asarray(args.history_len, dtype=np.int32),
            horizon=np.asarray(args.horizon, dtype=np.int32),
            include_heading_diff=np.asarray(1 if args.include_heading_diff else 0, dtype=np.int32),
            normalize_positions=np.asarray(1 if args.normalize_positions else 0, dtype=np.int32),
            v_max=np.asarray(args.v_max, dtype=np.float32),
            w_max=np.asarray(args.w_max, dtype=np.float32),
            workspace_scale=np.asarray(args.workspace_scale, dtype=np.float32),
            n_beams=np.asarray(n_beams, dtype=np.int32),
            input_dim=np.asarray(inputs_arr.shape[1], dtype=np.int32),
            output_dim=np.asarray(outputs_arr.shape[1], dtype=np.int32),
        )
        print(f"{split}: saved {dataset_out_path(split).resolve()}")

    print(
        f"Generating dataset with dt={args.dt}, history_len={args.history_len}, horizon={args.horizon}, "
        f"normalize_positions={args.normalize_positions}, include_heading_diff={args.include_heading_diff}"
    )
    print(f"Velocity norms: v_max={args.v_max}, w_max={args.w_max}; workspace_scale={args.workspace_scale}")

    run_split(train_list, "train")
    run_split(val_list, "val")
    run_split(test_list, "test")


if __name__ == "__main__":
    main()

