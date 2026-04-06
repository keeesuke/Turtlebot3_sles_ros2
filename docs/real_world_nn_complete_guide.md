# Real-World NN Planner — Complete Guide
## Data Collection → Training → Deployment → Validation

---

## Design Rationale

The NN planner is deployed **without a pre-built map**.  
At deployment time Cartographer starts fresh and builds the map from scratch while the robot navigates.

**Data collection must match this condition exactly:**

- Restart Cartographer at the beginning of **every** session → fresh map each time  
- Reposition obstacles between sessions → diverse training environments  
- Manual joystick driving → human demonstrations the NN will imitate  
- `turtlebot3_node` (hardware driver) stays running throughout — only Cartographer and the recorder restart

---

## Files Created / Modified

### New files

| File | Purpose |
|---|---|
| `src/turtlebot3_sles_data/turtlebot3_sles_data/robot_data_recorder_real_world.py` | ROS2 node: records TF2 pose, `/scan`, `/cmd_vel` to `session_*_REAL/` |
| `src/turtlebot3_sles_data/turtlebot3_sles_data/prepare_training_data_real_world.py` | Offline script: raw NPZ → `train/val/test_dataset.npz` |
| `docs/real_world_nn_pipeline.md` | Architecture reference |
| `docs/real_world_nn_complete_guide.md` | This file |

### Modified files

| File | Change |
|---|---|
| `src/turtlebot3_sles_data/CMakeLists.txt` | Added two new scripts to `install(PROGRAMS ...)` |
| `src/turtlebot3_sles_control/turtlebot3_sles_control/planner_nn_real_world.py` | Added `model_path` ROS parameter (load model from any path, no copy needed) |
| `src/turtlebot3_sles_control/launch/turtlebot3_planner_NN_real_world.launch.py` | Added `model_path` launch argument |
| `src/turtlebot3_sles_learning/turtlebot3_sles_learning/train_mlp.py` | Added `argparse` (`--data-dir`, `--save-dir`, `--epochs`, etc.) |

---

## Prerequisites

### Build (Linux / Docker, one time)

```bash
cd ~/Turtlebot3_sles_ros2
colcon build --symlink-install
source install/setup.bash
```

Add to `~/.bashrc` so you don't need to source manually:
```bash
echo "source ~/Turtlebot3_sles_ros2/install/setup.bash" >> ~/.bashrc
```

### Hardware driver (keep running the entire day)

Open a dedicated terminal and leave it running:

```bash
# Terminal A — run once, never kill
ros2 launch turtlebot3_bringup robot.launch.py
```

This publishes `/scan` and the TF chain `odom → base_footprint`.  
It does **not** need to restart between sessions.

---

## Phase 1: Data Collection

### Per-session routine (repeat ~100–300 times)

```
Step 1  Reposition obstacles in the arena

Step 2  Place the robot at the desired start position

Step 3  Kill the previous Cartographer (if still running)
        → Ctrl+C in Terminal B

Step 4  Start a fresh Cartographer
        Terminal B:
        ros2 launch turtlebot3_cartographer cartographer.launch.py

        Wait 2–3 s for the "map" TF to appear.

Step 5  Start the data recorder
        Terminal C:
        ros2 run turtlebot3_sles_data robot_data_recorder_real_world.py

        Confirm you see:
        "Press Ctrl+C to stop recording and save data."

Step 6  Start your joystick controller
        Terminal D:
        ros2 run <your_package> <your_joystick_node>
        (substitute with your actual joystick command)

Step 7  Drive the robot manually to the goal position

Step 8  Stop at the goal, then:
        → Ctrl+C in Terminal C  (recorder saves automatically)
        → Ctrl+C in Terminal D  (stop joystick)

        Confirm the save log:
        "✅ Data saved to: ~/robot_data/session_YYYYMMDD_HHMMSS_REAL"

Step 9  Quick check:
        ls -lt ~/robot_data/ | head -3
        (new session folder should be at the top)

Step 10 Go to Step 1
```

### Why Cartographer restarts every session

At deployment the NN planner runs with Cartographer starting from a **blank map**.  
The TF `map → base_footprint` is computed from this fresh SLAM state.  
If data collection uses a fully-warmed Cartographer with accumulated map, the TF quality at frame 0 would be better than at deployment, creating a train/test mismatch.  
Restarting Cartographer every session keeps both conditions identical.

### Session folder structure (after Ctrl+C)

```
~/robot_data/session_20260301_120000_REAL/
  ├── robot_states_20260301_120000.jsonl       # crash-safe incremental writes
  ├── control_inputs_20260301_120000.jsonl
  ├── lidar_scans_20260301_120000.jsonl
  ├── robot_states_20260301_120000.json        # JSON array (for tooling)
  ├── control_inputs_20260301_120000.json
  ├── lidar_scans_20260301_120000.json
  ├── robot_data_20260301_120000.npz           # raw NumPy (input to prepare script)
  ├── lidar_ranges_20260301_120000.npz         # separate LiDAR arrays
  └── session_info_20260301_120000.json        # metadata + start/end positions
```

### Timing per session

| Step | Duration |
|---|---|
| Reposition obstacles + place robot | ~20 s |
| Kill / restart Cartographer | ~5 s |
| Start recorder | ~2 s |
| Drive (30–90 s recommended) | 30–90 s |
| Ctrl+C + auto-save | ~5 s |
| Quick folder check | ~3 s |
| **Total per session** | **~1–2 min** |

### How many sessions to collect

| Stage | Sessions | Notes |
|---|---|---|
| First training run | 30–50 | Verify the pipeline works end-to-end |
| Meaningful model | 100–200 | Diverse obstacle placements |
| Production model | 200–500 | Various start/goal pairs |

---

## Phase 2: Preprocessing

Run this offline (Linux or macOS — no ROS2 required).

### Single session

```bash
python3 src/turtlebot3_sles_data/turtlebot3_sles_data/prepare_training_data_real_world.py \
    ~/robot_data/session_20260301_120000_REAL
```

Output:
```
~/robot_data/real_world_datasets/20260301_120000/
  ├── train_dataset.npz   (80%)
  ├── val_dataset.npz     (10%)
  ├── test_dataset.npz    (10%)
  └── dataset_info.json
```

### Merge all sessions into one dataset (recommended)

```bash
python3 src/turtlebot3_sles_data/turtlebot3_sles_data/prepare_training_data_real_world.py \
    ~/robot_data/session_*_REAL \
    --merge \
    --output-dir ~/robot_data/real_world_datasets/merged_run01
```

### Key options

| Option | Default | Meaning |
|---|---|---|
| `--lidar-max-range` | `1.0` | LiDAR clip value (m). **Must match the value used at inference.** |
| `--min-speed` | `0.01` | Filter frames where the robot was nearly stopped |
| `--lidar-sync-tolerance` | `0.15` | Max time difference (s) between state and LiDAR timestamps |
| `--split` | `0.8 0.1 0.1` | Train / Val / Test fractions |

### What the script does internally

1. Loads `robot_data_*.npz` + `lidar_ranges_*.npz` from each session  
2. **Post-hoc goal assignment** — the last recorded position becomes the goal for every frame in that session  
3. Transforms goal into robot frame for each timestep  
4. Nearest-neighbour sync: matches each state frame to the closest LiDAR scan (within tolerance)  
5. Normalises LiDAR: `inf / nan / ≤ 0 → 1.0 m`, clips, resamples to 360 rays  
6. Filters frames where the robot is nearly stopped  
7. Shuffles and splits → `train / val / test_dataset.npz`

### NPZ dataset format (consumed by `train_mlp.py`)

```
states           (N, 2)    [v (m/s), ω (rad/s)]
target_positions (N, 2)    goal in robot frame [x, y]
lidar_scans      (N, 360)  normalised LiDAR, range [0, 1.0 m]
control_linear   (N,)      v command recorded from /cmd_vel
control_angular  (N,)      ω command recorded from /cmd_vel
```

---

## Phase 3: Training

Training runs offline on Linux (GPU recommended but CPU works).

```bash
cd ~/robot_data/real_world_datasets/merged_run01

python3 ~/Turtlebot3_sles_ros2/src/turtlebot3_sles_learning/turtlebot3_sles_learning/train_mlp.py \
    --data-dir . \
    --save-dir ~/robot_data/real_world_models/run01 \
    --epochs 30
```

### CLI options

| Option | Default | Notes |
|---|---|---|
| `--data-dir` | `.` (current dir) | Folder with `train/val/test_dataset.npz` |
| `--save-dir` | `<data-dir>/real_world_models/` | Where `best_model.pth` is saved |
| `--epochs` | `20` | Increase to 30–50 for real-world data |
| `--batch-size` | `256` | |
| `--lr` | `1e-3` | |

### Output

```
~/robot_data/real_world_models/run01/
  ├── best_model.pth        ← deploy this
  └── training_curves.png   ← inspect loss curves
```

### Evaluation targets (test set)

| Metric | v_cmd | ω_cmd | Target |
|---|---|---|---|
| R² | — | — | ≥ 0.80 |
| MAE | — | — | v < 0.03 m/s, ω < 0.05 rad/s |
| Val loss plateau | — | — | Loss should stop decreasing by epoch 20–30 |

If R² < 0.6 → collect more sessions or check data quality.

### Iterative improvement loop

```
Collect more sessions
        ↓
Re-run prepare_training_data_real_world.py --merge (include old sessions)
        ↓
Re-train with --epochs 50
        ↓
Compare best_model.pth on test set
        ↓
Deploy and evaluate on robot
```

---

## Phase 4: Deployment

No model copying required. Pass the path directly at launch time.

### Build first (if code has changed)

```bash
cd ~/Turtlebot3_sles_ros2
colcon build --symlink-install
source install/setup.bash
```

### Start required stack

```bash
# Terminal A — hardware driver (may already be running)
ros2 launch turtlebot3_bringup robot.launch.py

# Terminal B — fresh Cartographer (fresh map, same as data collection)
ros2 launch turtlebot3_cartographer cartographer.launch.py

# Terminal C — RViz2 (optional, for 2D Goal Pose input and map visualisation)
ros2 launch turtlebot3_bringup rviz2.launch.py
```

### Launch NN planner

```bash
# Terminal D
ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py \
    model_path:=~/robot_data/real_world_models/run01/best_model.pth
```

The `model_path` argument accepts `~` expansion.  
The model file is read directly from that path — nothing is copied into the ROS install tree.

### How `model_path` works

```
model_path argument provided?
    YES → load from that absolute path (real-world trained model)
    NO  → fall back to best_model.pth next to the installed script
          (simulation model bundled with the package)
```

---

## Phase 5: Validation

### Startup logs to verify

```
NN model loaded from wrapped checkpoint: ~/robot_data/real_world_models/run01/best_model.pth
Real-world NN planner node started:
  - State source  : TF2 map→base_footprint (100 Hz sliding window)
  - LiDAR source  : /scan (real TurtleBot3 LDS-01)
  - Goal source   : /move_base_simple/goal (RViz2 2D Goal Pose)
  - Control rate  : 50 Hz NN inference
  - lidar_max_range clipped to 1.0 m (training range)
Waiting for TF2 (map→base_footprint) and /scan ...
```

### Set a goal

In RViz2: click **"2D Goal Pose"** button → click a point on the map.

This publishes to `/move_base_simple/goal` and the planner starts immediately:
```
New goal from RViz2: x=1.500 m, y=0.800 m, theta=0.0 deg
```

### Validation checklist

- [ ] Robot starts moving within 1–2 s of goal being set  
- [ ] Robot turns to face goal direction  
- [ ] Robot slows near obstacles (LiDAR-aware behaviour)  
- [ ] `Target reached! Stopping.` log appears when robot arrives  
- [ ] Trajectory plot saved to `~/robot_trajectory_nn_rw.png`  
- [ ] No `NN inference error` messages in the log  

### Stuck detection

If the robot makes no progress for 10 s, it stops and waits for a new goal:
```
Robot stuck for >10 s at (0.52, 1.23). Waiting for new goal.
```
Set a new goal from RViz2 to continue.

### Comparing models

To switch models without restarting the planner, kill Terminal D and relaunch with a different `model_path`:
```bash
ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py \
    model_path:=~/robot_data/real_world_models/run02/best_model.pth
```

---

## Quick Reference

### Terminal layout for a full day of data collection

```
Terminal A  ros2 launch turtlebot3_bringup robot.launch.py        [always on]
Terminal B  ros2 launch turtlebot3_cartographer cartographer.launch.py  [restart each session]
Terminal C  ros2 run turtlebot3_sles_data robot_data_recorder_real_world.py  [restart each session]
Terminal D  ros2 run <your_package> <your_joystick_node>           [restart each session]
```

### Per-session command sequence (copy-paste ready)

```bash
# Kill previous Cartographer (Terminal B)
Ctrl+C

# Fresh Cartographer (Terminal B)
ros2 launch turtlebot3_cartographer cartographer.launch.py

# Start recorder (Terminal C)
ros2 run turtlebot3_sles_data robot_data_recorder_real_world.py

# Start joystick (Terminal D)
ros2 run <your_package> <your_joystick_node>

# ... drive ... then stop and:
Ctrl+C   # Terminal D (joystick)
Ctrl+C   # Terminal C (recorder) — data saves automatically

# Confirm new session
ls -lt ~/robot_data/ | head -3
```

### Full pipeline commands (offline, run once after collection)

```bash
# Preprocess all sessions
python3 src/turtlebot3_sles_data/turtlebot3_sles_data/prepare_training_data_real_world.py \
    ~/robot_data/session_*_REAL --merge \
    --output-dir ~/robot_data/real_world_datasets/merged_run01

# Train
python3 src/turtlebot3_sles_learning/turtlebot3_sles_learning/train_mlp.py \
    --data-dir ~/robot_data/real_world_datasets/merged_run01 \
    --save-dir ~/robot_data/real_world_models/run01 \
    --epochs 30

# Deploy
ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py \
    model_path:=~/robot_data/real_world_models/run01/best_model.pth
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Recorder shows `0 state frames` | Cartographer not running / TF not ready | Wait 3 s after Cartographer starts before launching recorder |
| `NN model not found` at launch | Wrong path or typo | `ls ~/robot_data/real_world_models/run01/` to verify |
| Robot does not move after goal set | `state_ready` or `lidar_ready` still False | Check Cartographer and `/scan` are active |
| R² < 0.6 after training | Too few sessions or low diversity | Collect more sessions; vary start/goal and obstacle layouts |
| Valid frame ratio < 70% | Too much stopped time | Drive continuously; stop only at the very end of each session |
| Stuck detection triggers immediately | Map frame not stable at start | Wait 3–5 s for Cartographer to initialise before driving |
