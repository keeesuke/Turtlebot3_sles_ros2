# Real-World MPC+NN Switching Planner

## Overview

The MPC+NN switching planner (`planner_switch_mpc_nn_real_world.py`) implements a **simplex architecture** that combines two controllers:

- **HPA (High-Performance Autonomous)** — Neural network for fast, reactive navigation at higher speeds.
- **HAA (High-Assurance Autonomous)** — MPPI-based Model Predictive Control for safety-guaranteed navigation at conservative speeds.

The robot starts in HPA (NN) mode for performance. Every planning cycle, a **feasibility check** verifies that MPC could still recover from where the NN is heading. If not, the system pre-emptively switches to MPC before anything goes wrong. Once MPC stabilises the robot in a safe region, NN takes control again.

---

## Files

| File | Purpose |
|------|---------|
| `turtlebot3_sles_control/planner_switch_mpc_nn_real_world.py` | Main controller node |
| `launch/turtlebot3_planner_switch_MPC_NN_real_world.launch.py` | Launch file with tuned parameters |
| `CMakeLists.txt` | Updated to install the new executable |

### What was modified vs existing code

The new controller was built by merging three existing files:

| Source file | What was taken |
|-------------|---------------|
| `planner_haa_real_world.py` | **Base**: MPPI class, MPPIPlanner, OccupancyGridMap, KanayamaController, TF2/odom state estimation, /map subscription, inflation map publisher, stuck detection, EMA+rate-limiter command filtering, debug visualisation |
| `planner_nn_real_world.py` | MLP class, NN model loading (3 checkpoint formats, `model_path` parameter), `/scan` subscription with Best-Effort QoS, real LiDAR processing (`inf`/`nan` handling, `lidar_max_range` clipping) |
| `planner_switch_mpc_nn.py` | Switching state machine (HPA/HAA logic), `get_nn_control()`, `generate_braking_trajectory()`, feasibility check, safety margin recovery check, switch point tracking |

### Key changes from the simulation switching controller

The original `planner_switch_mpc_nn.py` was simulation-only. The following were updated to match the real-world codebase:

**Infrastructure:**
- State source: Gazebo `/gazebo/model_states` -> TF2 `map->base_footprint` (100 Hz) + `/odom` with EMA smoothing (alpha=0.3)
- Map source: `/lidar_occupancy_map` -> `/map` from Cartographer SLAM
- LiDAR: `/simulated_scan` with `-1 -> 1.0` -> `/scan` with Best-Effort QoS, `inf`/`nan -> lidar_max_range`
- Control loop: 20 Hz -> 50 Hz
- Goal: Fixed ROS parameter -> `/move_base_simple/goal` (RViz2 interactive)
- On failure: `rclpy.shutdown()` -> Reset and wait for new goal (node stays alive)
- Added: 100 Hz TF2 state update, stuck detection (10 s timeout), EMA + rate-limiter on commands, `/inflation_map` and `/slam_pose` publishers

**MPPI parameters:**

| Parameter | Old (sim) | New (real-world) |
|-----------|-----------|-----------------|
| sigma | 1 | 5 |
| temperature | 0.1 | 1.0 |
| num_rollouts | 1000 | 1500 |
| noise ramp direction | Decreasing (1 -> 1/N) | Increasing (1/N -> 1) |
| obstacle threshold | > 0.1 | > 50 |
| near-obstacle penalty | x2.0 | x5 |
| cold start angular accel | 0.5 x alpha_max | 0.3 x alpha_max |
| smoothing_weight | 0.0 | 0.1 |
| MPPIPlanner logger | None | `self.get_logger()` |

**NN model loading:**
- Hardcoded path -> `model_path` ROS parameter with fallback
- 2 checkpoint formats -> 3 (wrapped dict, raw state dict, full model)

---

## Switching Logic

### State machine

```
                     feasibility OK
        +-------------------------------------+
        |                                     |
        v                                     |
   +---------+    feasibility FAIL    +-------------+    safety margin OK    +---------+
   |   HPA   | --------------------> |   BRAKING   | --------------------> |   HAA   |
   |  (NN)   |                       | (2 loops)   |                       |  (MPC)  |
   +---------+                       +-------------+                       +---------+
        ^                                                                       |
        |                        safety margin satisfied                        |
        +-----------------------------------------------------------------------+
```

### While in HPA (NN mode) — 10 Hz planning loop

1. **Get NN output**: Query neural network with `[v, w, goal_robot, lidar_360]` -> `(v_cmd, w_cmd)`
2. **One-step prediction**: Simulate where the robot would be after applying the NN command for one timestep
3. **Braking simulation**: From that predicted state, apply maximum braking for `N_r = 3` steps to get the "worst-case recoverable state"
4. **HAA feasibility check**: Run the full MPPI planner from the recoverable state. If a valid 4-second trajectory to the goal exists, the NN command is safe.
5. **Decision**:
   - **Feasible**: Stay in HPA. Control loop publishes NN output directly at 50 Hz.
   - **Not feasible**: Enter 2-loop braking phase, then switch to HAA.

### Braking phase

When feasibility fails, the robot must decelerate from HPA speed (up to 0.26 m/s) to a speed HAA can handle (0.2 m/s). This is done by generating a max-braking trajectory (`a = -a_limit * 1.2`) and tracking it with the Kanayama controller for 2 planning loops (0.2 s). After braking completes, the planner switches to HAA.

### While in HAA (MPC mode)

1. Clip current velocity to HAA limits
2. Run MPPI planner (1500 rollouts, 40-step horizon = 4 seconds)
3. Track the trajectory with the Kanayama controller + EMA + rate-limiter at 50 Hz
4. **Safety margin check**: If velocity is low enough AND the robot position has sufficient clearance (no obstacle within `N_s * dt * v_limit + v^2 / (2a) + 0.12` metres), switch back to HPA

### Control loop (50 Hz)

The control loop branches based on the active planner:

| Mode | Behaviour |
|------|-----------|
| HPA (NN, no trajectory) | Run NN inference, publish `(v, w)` directly |
| HPA (braking, with trajectory) | Kanayama tracking + EMA + rate-limiter |
| HAA (MPC, with trajectory) | Kanayama tracking + EMA + rate-limiter |

---

## Parameters

### Launch file defaults

```python
# MPC/HAA limits (conservative)
horizon_haa     = 40      # 4 s prediction horizon
dt              = 0.1     # planning timestep
v_limit_haa     = 0.2     # m/s
omega_limit_haa = 0.9     # rad/s
a_limit         = 0.5     # m/s^2
alpha_limit     = 0.5     # rad/s^2
robot_radius    = 0.15    # m

# NN/HPA limits (wider)
v_limit_hpa     = 0.26    # m/s
omega_limit_hpa = 1.82    # rad/s

# LiDAR
lidar_max_range = 1.0     # m (must match training)

# Kanayama tracking
kx = 0.6, ky = 8.0, kth = 1.6, kv = 1.0, kw = 1.0
```

---

## Output Plots

On goal-reached the node saves a single plot: `~/robot_trajectory_switch.png`.

### `~/robot_trajectory_switch.png` — Trajectory + switching plot

2×3 subplot layout. Every subplot is coloured by the active planner at each sample so switching timing is visible as a colour change in the line itself (no separate switch-point markers):

- **HPA (NN)** — blue `#2196F3`
- **HAA (MPC)** — red `#E53935`
- **Braking** — orange `#FF9800` (last ~2 planning loops of an HPA segment that ends in a switch to HAA)

Panel layout:

| Position | Content |
|---|---|
| Top-left (ax1) | Robot position trajectory over inflated occupancy map, coloured by planner. Start (green), goal (red star), robot-radius tube. |
| Top-middle (ax2) | Unwrapped orientation θ vs time, coloured by planner. |
| Top-right (ax3) | Odom-measured **linear velocity** vs time, coloured by planner. Dashed HAA limit (0.2 m/s) and dotted HPA limit (0.26 m/s) reference lines. |
| Bottom-left (ax4) | Odom-measured **angular velocity** vs time, coloured by planner. HAA/HPA ± limit reference lines. |
| Bottom-middle (ax5) | Published `/cmd_vel` **linear velocity command** vs time, coloured by planner at command time. |
| Bottom-right (ax6) | Published `/cmd_vel` **angular velocity command** vs time, coloured by planner at command time. |

Note the distinction between ax3 (odom — what the robot actually did) and ax5 (command — what we asked it to do): on hardware these diverge due to motor time constant, battery voltage sag, and Waffle Pi's practical top speed (~0.22 m/s) falling below the nominal 0.26 m/s spec.

---

## Usage

```bash
# Basic launch (model loaded from install tree)
ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_NN_real_world.launch.py

# With explicit model path
ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_NN_real_world.launch.py \
    model_path:=/home/acrl/robot_data/real_world_models/run01/best_model.pth
```

Prerequisites (must already be running):
1. `turtlebot3_node` (hardware driver — `/scan`, `/odom`, `/tf odom->base_footprint`)
2. `turtlebot3_cartographer` (SLAM — `/map`, `/tf map->odom`)
3. `rviz2` (optional — for interactive goal via '2D Goal Pose')

Set a goal using the '2D Goal Pose' button in RViz2. The robot starts in HPA (NN) mode. Monitor switching via the log messages:

```
[INFO] HAA check feasible: True       # NN is safe, staying in HPA
[INFO] HAA check feasible: False      # NN heading to unsafe region
[INFO] Braking mode: loop 1/2         # Decelerating
[INFO] Braking phase complete — switching to HAA
[INFO] Safety margin satisfied — switching back to HPA (NN)
```

---

## Architecture Diagram

```
  /scan (LiDAR)          /map (Cartographer)       /odom (velocity)
       |                        |                        |
       v                        v                        v
  +-----------+          +-------------+          +-------------+
  | lidar_cb  |          |   map_cb    |          |   odom_cb   |
  | (360 rays)|          | (OccupancyGrid)        | (EMA filter)|
  +-----------+          +-------------+          +-------------+
       |                        |                        |
       +----------+-------------+------------------------+
                  |
                  v
         +----------------+
         | planning_loop  |  (10 Hz)
         |   HPA or HAA   |
         +-------+--------+
                 |
    +------------+------------+
    |                         |
    v                         v
+--------+              +---------+
|  HPA   |              |   HAA   |
| get_nn |              |  MPPI   |
| control|              | planner |
+--------+              +---------+
    |                         |
    +------------+------------+
                 |
                 v
         +----------------+
         | control_loop   |  (50 Hz)
         | Kanayama / NN  |
         +-------+--------+
                 |
                 v
            /cmd_vel
```
