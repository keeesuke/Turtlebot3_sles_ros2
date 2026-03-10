#!/usr/bin/env bash
# =============================================================================
# launch_simulation_planner.sh
# ROS2 equivalent of launch_simulation_HPA.sh / launch_simulation_MPC.sh
#
# Launches the full SLES simulation pipeline using the HAA/MPPI planner:
#   1. Gazebo world with random obstacles (turtlebot3_sles_worlds)
#   2. LiDAR simulation publisher   (turtlebot3_sles_perception)
#   3. LiDAR occupancy map node     (turtlebot3_sles_perception)
#   4. HAA/MPPI planner node        (turtlebot3_sles_control)
#
# Usage:
#   bash launch_simulation_planner.sh [GOAL] [SEED]
#
#   GOAL  — optional, default "[-1.5, -1.5, 0, 0, 0]"
#   SEED  — optional random seed for obstacle placement (leave empty for random)
#
# Example:
#   bash launch_simulation_planner.sh "[-1.5, -1.5, 0, 0, 0]" 42
# =============================================================================
set -e

GOAL="${1:-[-1.5, -1.5, 0, 0, 0]}"
SEED="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "${SCRIPT_DIR}")"

# Source environment
source "${WS_DIR}/sles_env.sh"

echo "=== SLES Simulation: HAA/MPPI Planner ==="
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

# Wait for Gazebo to finish loading before starting perception nodes
echo "      Waiting 20s for Gazebo to fully initialize..."
sleep 20

# ── Step 2: Launch LiDAR simulation (background) ─────────────────────────────
# Note: simulate_lidar_publisher_new is already launched inside the world launch
# file. The mapping node is launched separately here.

# ── Step 3: Launch occupancy map node (background) ───────────────────────────
echo "[3/4] Launching LiDAR occupancy mapping node..."
ros2 launch turtlebot3_sles_perception turtlebot3_simulate_mapping.launch.py &
MAPPING_PID=$!
echo "      Mapping PID: ${MAPPING_PID}"

echo "      Waiting 3s for mapping node to initialize..."
sleep 3

# ── Step 4: Launch HAA planner (foreground — keeps terminal alive) ────────────
echo "[4/4] Launching HAA/MPPI planner..."
echo "      (Press Ctrl+C to stop all nodes)"
ros2 launch turtlebot3_sles_control turtlebot3_planner.launch.py \
    goal:="${GOAL}"

# Cleanup on exit
trap "kill ${GAZEBO_PID} ${MAPPING_PID} 2>/dev/null || true" EXIT
