# Real-World Training Data Collection SOP

Step-by-step guide for collecting imitation learning data on the TurtleBot3 Waffle Pi using joystick teleoperation.

## Prerequisites

- TurtleBot3 Waffle Pi powered on and connected to the same network
- Logitech F710 wireless gamepad (USB dongle plugged into the Pi)
- Obstacles placed in the environment

## Setup (4 SSH terminals + 2 local terminals)

Every local (non-SSH) terminal needs the environment sourced first:

```bash
. ~/sles/Turtlebot3_sles_ros2/sles_env.sh
```

### Terminal 1 (SSH) — Bringup

```bash
ssh sigrobotics@10.195.154.171   # password: sigrobotics
. ~/.bashrc
ros2 launch turtlebot3_bringup robot.launch.py
```

### Terminal 2 (SSH) — Joy Node

```bash
ssh sigrobotics@10.195.154.171
. ~/.bashrc
ros2 run joy joy_node
```

### Terminal 3 (SSH) — Joy-to-CmdVel Bridge

```bash
ssh sigrobotics@10.195.154.171
. ~/.bashrc
cd ~/sles/joy_controller
python3 joy_teleop_bridge.py
```

**Activate the joystick now:**
1. Press the center **Logitech** button on the gamepad to wake it up
2. Set the back switch to **X** (not D)
3. Verify: `ros2 topic echo /joy` should show values when you move the sticks

### Terminal 4 (local) — Cartographer SLAM

Place the robot at your desired **start position** first, then launch SLAM:

```bash
ros2 launch turtlebot3_cartographer cartographer.launch.py \
    use_sim_time:=false resolution:=0.02 publish_period_sec:=0.01
```

> Launch SLAM *after* placing the robot. If you launch first and then move the robot to the start position, the SLAM map will have extra accuracy from that initial traversal, which won't be available in deployment.

### Terminal 5 (local) — Data Recorder

```bash
ros2 run turtlebot3_sles_data robot_data_recorder_real_world.py
```

## Collect Data

1. Drive the robot with the joystick (left stick = forward/back, right stick = turn)
2. Navigate around obstacles as you want the robot to learn
3. Press **Ctrl+C** to stop recording

Check the output — you should see something like:

```
Recorded — states: 2724, controls: 1409, lidar: 341
```

**All three counts must be non-zero.** If any is 0, that data stream failed (check that bringup, SLAM, and joystick are all running).

Data is saved to `~/robot_data/session_YYYYMMDD_HHMMSS_REAL/`.

## Collect More Episodes

You do **not** need to restart bringup, joy node, or the bridge. Just:

1. Relocate the robot to a new start position
2. Rearrange obstacles if desired
3. **Ctrl+C** the Cartographer terminal and re-launch it (fresh map each episode)
4. Re-run the data recorder (Terminal 5)
5. Drive, Ctrl+C, confirm data saved
6. Repeat

---

## (Optional) Train and Deploy

If you want to see the result of your collected data:

### Pre-process

Merges all sessions, automatically extracts start/goal positions from each episode (no manual labeling needed), syncs LiDAR/state timestamps, and outputs train/val/test splits.

```bash
conda activate base
cd ~/sles/Turtlebot3_sles_ros2
python3 src/turtlebot3_sles_data/turtlebot3_sles_data/prepare_training_data_real_world.py \
    ~/robot_data/session_*_REAL --merge \
    --output-dir ~/robot_data/real_world_datasets/merged_run01
```

### Train

Trains a 3-layer MLP (input: velocity + goal + 360 LiDAR rays -> output: v, omega commands). Requires PyTorch (installed in conda `base`).

```bash
conda activate base
cd ~/robot_data/real_world_datasets/merged_run01
python3 ~/sles/Turtlebot3_sles_ros2/src/turtlebot3_sles_learning/turtlebot3_sles_learning/train_mlp.py \
    --data-dir . \
    --save-dir ~/robot_data/real_world_models/run01 \
    --epochs 30
```

### Deploy

Launches the NN planner. Set a goal using RViz2 **2D Goal Pose** button.
Make sure to close any node that affect /cmd_vel including joy stick node, otherwise joy stick or teleop keyboard keeps interrupting NN-based control. 

```bash
conda activate base
ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py \
    model_path:=~/robot_data/real_world_models/run01/best_model.pth
```
