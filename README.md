# Turtlebot3_sles_ros2

## Quick Start

### 1) Clone the repository (with submodules)

```bash
git clone --recursive <your-repo-url>
cd Turtlebot3_sles_ros2
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

### 2) Open a new terminal and set up the environment

Edit `sles_env.sh` first and set `SLES_WS` to your local workspace path.

```bash
source /path/to/Turtlebot3_sles_ros2/sles_env.sh
```

On the first run only, enable execute permissions for launch/scripts in `sles_env.sh`.
Those `chmod +x ...` lines are currently commented out; uncomment them once, run `source sles_env.sh`, then you can comment them back if you want.

### 3) Build all packages

```bash
colcon build
```

After build:

```bash
source install/setup.bash
```

### 4) Launch the simulation stack

You can launch everything from shell scripts under `scripts/`:

```bash
bash scripts/launch_simulation_planner.sh
bash scripts/launch_simulation_planner_nn.sh
bash scripts/launch_simulation_diffusion.sh
```

You can also launch each ROS 2 launch file manually (world/perception/control) if you prefer per-node control.

## Notes

- Run `source sles_env.sh` in every new terminal before `ros2 launch`.
