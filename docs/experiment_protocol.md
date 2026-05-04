# Experiment Protocol — MPC / NN / Switching

How to run, record, and report a navigation experiment with each of the three control logics for the report/paper.

## Output layout

Each run produces one folder under `~/robot_data/experiments/`:

```
~/robot_data/experiments/
├── 20260427_153012_mpc/        ← one run, MPC-only logic
│   ├── robot_states_<ts>.{jsonl,json}    — pose + odom twist time-series
│   ├── control_inputs_<ts>.{jsonl,json}  — /cmd_vel time-series
│   ├── lidar_scans_<ts>.{jsonl,json}     — /scan time-series
│   ├── robot_data_<ts>.npz               — merged numpy arrays
│   ├── lidar_ranges_<ts>.npz             — raw scan arrays
│   ├── session_info_<ts>.json            — counts + start/end position
│   ├── experiment_summary.json           — run-level metrics + flags
│   └── robot_trajectory.png              — plot saved by the planner
├── 20260427_153512_nn/         ← one run, NN-only
│   └── …  (incl. robot_trajectory_nn_rw.png)
├── 20260427_154045_switch/     ← one run, Switching
│   └── …  (incl. robot_trajectory_switch.png)
└── .current_run                ← internal pointer, exists only while a recorder is running
```

The recorder always writes to `<TIMESTAMP>_<LOGIC>` so multiple runs of the same logic do not overwrite each other.

## Setup (same as data collection SOP)

Prereqs already running on the robot + workstation:
1. `ros2 launch turtlebot3_bringup robot.launch.py`
2. `ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=false resolution:=0.02 publish_period_sec:=0.01`
3. `rviz2`
4. **Joy node and bridge must be CLOSED** for NN-only and Switching runs (they write `/cmd_vel`).

## Per-run procedure

Pick the logic for this run, then run **two terminals**:

### Terminal A — Experiment recorder

```bash
ros2 run turtlebot3_sles_data experiment_recorder.py --logic <mpc|nn|switch>
```

The recorder creates `~/robot_data/experiments/<ts>_<logic>/` and writes `~/robot_data/experiments/.current_run` so the planner's plot saves directly into the run folder.

### Terminal B — Planner (one of three)

| Logic | Launch command |
|---|---|
| MPC only | `ros2 launch turtlebot3_sles_control turtlebot3_planner_real_world.launch.py` |
| NN only | `ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py model_path:=~/robot_data/real_world_models/run01/best_model.pth` |
| Switching | `ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_NN_real_world.launch.py model_path:=~/robot_data/real_world_models/run01/best_model.pth` |

### Run

1. Set a goal in RViz2 (**2D Goal Pose** button).
2. Wait for the robot to reach the goal (the planner saves the trajectory plot into the run folder).
3. **Ctrl+C the recorder (Terminal A)** to finalise data dump and write `experiment_summary.json`.
4. Stop the planner (Ctrl+C in Terminal B).

Repeat from Terminal A for each new run.

## Files saved

| File | Format | Contents |
|---|---|---|
| `robot_states_<ts>.jsonl` | JSON-lines | `{timestamp, position[x,y,0], orientation[qx,qy,qz,qw], yaw, linear_velocity, angular_velocity}` per row, ≤50 Hz |
| `control_inputs_<ts>.jsonl` | JSON-lines | `{timestamp, linear_x..., angular_z}` per `/cmd_vel` publish |
| `lidar_scans_<ts>.jsonl` | JSON-lines | `{timestamp, ranges[N], angle_*, range_*}` ≤10 Hz |
| `robot_data_<ts>.npz` | NumPy | merged arrays: `positions`, `orientations`, `linear_velocities`, `angular_velocities`, `control_linear`, `control_angular`, `*_timestamps`, `lidar_*` |
| `lidar_ranges_<ts>.npz` | NumPy | `scan_0..scan_{N-1}` raw arrays (one per recorded scan) |
| `session_info_<ts>.json` | JSON | counts + start/end position |
| `experiment_summary.json` | JSON | metrics — see below |
| `robot_trajectory*.png` | PNG | 2×3 plot saved by the planner at goal-reach |

## Analysis window

All summary metrics and the saved `*.json` / `robot_data_<ts>.npz` / `lidar_ranges_<ts>.npz` files cover only the **active control window**:

```
window_start = timestamp of first /cmd_vel message published
window_end   = timestamp the robot first enters the goal tolerance (0.10 m)
               (or recorder Ctrl+C time if goal was not reached)
```

Setup time before the goal/cmd starts and idle time after goal-reach are excluded from saved JSON/NPZ and from all derived metrics. The raw `*.jsonl` files are NOT filtered — they keep the full unwindowed stream as a debugging fallback.

## `experiment_summary.json` fields

```jsonc
{
  "logic": "mpc | nn | switch",
  "timestamp": "20260427_153012",
  "run_folder": "/home/acrl/robot_data/experiments/20260427_153012_mpc",
  "analysis_window": {
    "first_cmd_vel_ts": 1777315750.42,   // window start (first /cmd_vel)
    "goal_reach_ts":    1777315768.76,   // window end   (or null if not reached)
    "window_start_ts":  1777315750.42,
    "window_end_ts":    1777315768.76,
    "note": "Metrics & saved JSON/NPZ cover this window only. Raw .jsonl files contain the full unwindowed stream."
  },
  "duration_sec": 18.34,                 // window_end - window_start
  "travel_distance_m": 3.61,             // sum of SLAM-TF pose diffs INSIDE
                                         // window. Inflated by localisation
                                         // jitter — keep for reference only.
  "avg_speed_m_s": 0.197,                // distance / duration. Inherits the
                                         // jitter inflation above and can
                                         // exceed the planner's velocity cap.
                                         // For the paper, use
                                         // cmd_linear_velocity.mean instead
                                         // (analyze_experiments.py does this
                                         // automatically).
  "goal_position": [3.5, -0.5],          // first goal published in this run, or null
  "goal_reached": true,                  // first time inside 0.10 m of goal
  "final_distance_to_goal_m": 0.08,
  "samples": {                           // counts AFTER windowing
    "states":       917,
    "control_cmds": 459,
    "lidar_scans":  178
  },
  "odom_linear_velocity":  {"mean": 0.197, "min": 0.0, "max": 0.241, "std": 0.073},
  "odom_angular_velocity": {"mean": -0.011, "min": -0.948, "max": 0.875, "std": 0.291},
  "cmd_linear_velocity":   {"mean": 0.247, "min": 0.0, "max": 0.260, "std": 0.034},
  "cmd_angular_velocity":  {"mean": -0.020, "min": -1.820, "max": 1.815, "std": 0.421}
}
```

## Parameters used by each logic

These are the parameters in the corresponding launch file (= what to cite in the paper).

### Common
| Param | Value | Meaning |
|---|---|---|
| `dt` | 0.1 s | Planning loop period (10 Hz) |
| `robot_radius` | 0.15 m | Footprint with safety margin |
| `a_limit` | 0.5 m/s² | Linear acceleration limit |
| `alpha_limit` | 0.5 rad/s² | Angular acceleration limit |

### MPC only — `turtlebot3_planner_real_world.launch.py`
| Param | Value | Meaning |
|---|---|---|
| `horizon_haa` | 40 | MPPI rollout horizon (4 s @ 10 Hz) |
| `v_limit_haa` | 0.2 m/s | MPC linear-velocity cap |
| `omega_limit_haa` | 0.9 rad/s | MPC angular-velocity cap |
| `kx, ky, kth, kv, kw` | 0.6, 8.0, 1.6, 1.0, 1.0 | Kanayama tracking gains |
| `kix, kiy, kith` | 0.1, 0.0, 0.1 | Kanayama integral gains |
| `max_integral` | 1.0 | Integral wind-up clamp |

### NN only — `turtlebot3_planner_NN_real_world.launch.py`
| Param | Current value | Meaning |
|---|---|---|
| `v_limit_haa` | 0.2 m/s | Velocity cap (clip before publish) |
| `omega_limit_haa` | 1.0 rad/s | Angular cap (clip before publish) |
| `lidar_max_range` | 1.0 m | LiDAR clip range — must match training |
| `model_path` | `…/best_model.pth` | NN weights file |

### Switching (HPA=NN, HAA=MPC) — `turtlebot3_planner_switch_MPC_NN_real_world.launch.py`
| Param | Current value | Meaning |
|---|---|---|
| `horizon_haa` | 40 | MPC rollout horizon |
| `dt` | 0.1 s | Planning period |
| `v_limit_haa` | 0.14 m/s | MPC (HAA) velocity cap (lowered from 0.2 to widen HPA-HAA gap) |
| `omega_limit_haa` | 0.9 rad/s | MPC (HAA) angular cap |
| `v_limit_hpa` | 0.20 m/s | NN (HPA) velocity cap (lowered from 0.26 to match measured hardware cap) |
| `omega_limit_hpa` | 1.0 rad/s | NN (HPA) angular cap |
| `a_limit, alpha_limit` | 0.5 / 0.5 | Acceleration limits |
| `robot_radius` | 0.15 m | Footprint |
| `lidar_max_range` | 1.0 m | LiDAR clip range |
| `kx … max_integral` | (same as MPC-only) | Kanayama tracking + integral gains |
| `model_path` | `…/best_model.pth` | NN weights for HPA mode |

> The `v_limit_*` and `omega_limit_*` values may change as the policy is tuned. Always cite the values from the launch file at the time of the run rather than these tables.

## Suggested per-logic metric table for the paper

For each logic, run N≥3 trials with the same start and goal, then report:

| Metric | MPC | NN | Switching |
|---|---|---|---|
| Goal-reach rate (success / N) | | | |
| Mean travel time (s) | mean ± std of `duration_sec` | | |
| Mean travel distance (m) | mean ± std of `travel_distance_m` (note: jitter-inflated) | | |
| Mean commanded v (m/s) | mean ± std of `cmd_linear_velocity.mean` | | |
| Peak commanded v (m/s) | mean of `cmd_linear_velocity.max` | | |
| Mean measured v (m/s) | mean of `odom_linear_velocity.mean` | | |
| Peak measured v (m/s) | mean of `odom_linear_velocity.max` | | |
| Number of switches | — | — | from plot's planner-history segments |

`analyze_experiments.py` builds this table automatically — see the next section. The script intentionally **does not** use `avg_speed_m_s` from the recorder summary (= path / duration), because the SLAM TF positions are noisy and `np.diff().sum()` accumulates that noise into a fake +50% travel distance, pushing the resulting "speed" above the planner's velocity cap. Reporting `cmd_linear_velocity.mean` (or `odom_linear_velocity.mean`) is both bounded and physically meaningful.

## Aggregate analysis (paper-ready performance matrix)

`analyze_experiments.py` walks every `<TS>_<LOGIC>/experiment_summary.json` under `~/robot_data/experiments/` and produces:

| Output | Purpose |
|---|---|
| `runs.csv` | One row per run — drop into spreadsheets / Pandas. |
| `by_logic.csv` | Aggregated mean/std/min/max for every metric. |
| `by_logic.md` | Markdown table ready to paste into the paper. |
| `metrics_comparison.png` | Bar chart with error bars (success rate, time, distance, speed, peak v). |
| `trajectories.png` | Overlay of every run's trajectory, coloured by logic. |

Run with no arguments to analyse everything; outputs go to `~/robot_data/experiments/analysis_<NOW>/`.

```bash
# All runs, default output folder
ros2 run turtlebot3_sles_data analyze_experiments.py

# Limit to a subset (e.g. paper-ready set)
ros2 run turtlebot3_sles_data analyze_experiments.py \
    --data-dir ~/robot_data/experiments/for_paper

# Time / distance / speed averaged over successful runs only
# (failed runs hit Ctrl+C and skew duration)
ros2 run turtlebot3_sles_data analyze_experiments.py --successful-only-for-time

# Only one logic
ros2 run turtlebot3_sles_data analyze_experiments.py --logic switch
```

The markdown table is also printed to stdout immediately so you can sanity-check before opening the file.

## Re-analysing later (training-data pipeline)

`prepare_training_data_real_world.py` accepts these run folders (they share the same NPZ schema as recording sessions). E.g., to re-process all NN-only runs into a unified dataset:

```bash
python3 src/turtlebot3_sles_data/turtlebot3_sles_data/prepare_training_data_real_world.py \
    ~/robot_data/experiments/*_nn --merge \
    --output-dir /tmp/nn_runs_dataset
```
