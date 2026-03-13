#!/usr/bin/env bash
# =============================================================================
# launch_simulation_planner_nn.sh
# ROS2 equivalent of launch_simulation_NN.sh
#
# Launches the full SLES simulation pipeline using the Neural Network planner:
#   1. Gazebo world with random obstacles (turtlebot3_sles_worlds)
#   2. LiDAR simulation publisher   (turtlebot3_sles_perception)
#   3. LiDAR occupancy map node     (turtlebot3_sles_perception)
#   4. NN planner node              (turtlebot3_sles_control)
#
# Usage:
#   bash launch_simulation_planner_nn.sh [GOAL] [SEED]
#
#   GOAL  — optional, default "[-1.5, -1.5, 0, 0, 0]"
#   SEED  — optional random seed for obstacle placement (leave empty for random)
#
# Example:
#   bash launch_simulation_planner_nn.sh "[-1.5, -1.5, 0, 0, 0]" 42
# =============================================================================
set -e

GOAL="${1:-[-1.5, -1.5, 0, 0, 0]}"
SEED="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "${SCRIPT_DIR}")"

# Source environment
source "${WS_DIR}/sles_env.sh"

echo "=== SLES Simulation: Neural Network Planner ==="
echo "    Goal : ${GOAL}"
echo "    Seed : ${SEED:-<random>}"
echo ""

# ── Step 1: Launch Gazebo world (background) ──────────────────────────────────
echo "[1/4] Launching Gazebo world..."
if [ -n "${SEED}" ]; then
    ros2 launch turtlebot3_sles_worlds turtlebot3_custom_world_random.launch.py \
        seed:="${SEED}" &
else
    ros2 launch turtlebot3_sles_worlds turtlebot3_custom_world_random.launch.py &
fi
GAZEBO_PID=$!
echo "      Gazebo PID: ${GAZEBO_PID}"

echo "      Waiting 20s for Gazebo to fully initialize..."
sleep 20

# ── Step 2: Launch LiDAR simulation (background) ─────────────────────────────
echo "[2/4] Launching LiDAR simulation publisher..."
ros2 launch turtlebot3_sles_perception turtlebot3_simulate_lidar_random.launch.py &
LIDAR_PID=$!
echo "      LiDAR PID: ${LIDAR_PID}"


# ── Step 3: Launch occupancy map node (background) ───────────────────────────
echo "[3/4] Launching LiDAR occupancy mapping node..."
ros2 launch turtlebot3_sles_perception turtlebot3_simulate_mapping.launch.py &
MAPPING_PID=$!
echo "      Mapping PID: ${MAPPING_PID}"

echo "      Waiting 3s for mapping node to initialize..."
sleep 3

# ── Step 4: Launch NN planner (foreground) ────────────────────────────────────
echo "[4/4] Launching Neural Network planner..."
echo "      (Press Ctrl+C to stop all nodes)"
ros2 launch turtlebot3_sles_control turtlebot3_planner_NN.launch.py \
    goal:="${GOAL}"

# Cleanup on exit
trap "kill ${GAZEBO_PID} ${MAPPING_PID} 2>/dev/null || true" EXIT
