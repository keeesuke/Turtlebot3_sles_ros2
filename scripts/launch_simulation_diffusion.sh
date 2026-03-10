#!/usr/bin/env bash
# =============================================================================
# launch_simulation_diffusion.sh
# ROS2 equivalent of launch_simulation_diffusion.sh (ROS1)
#
# Launches the full SLES simulation pipeline using the Diffusion Policy planner:
#   1. Gazebo world with random obstacles (turtlebot3_sles_worlds)
#   2. LiDAR simulation publisher   (turtlebot3_sles_perception)
#   3. LiDAR occupancy map node     (turtlebot3_sles_perception)
#   4. Diffusion planner node       (turtlebot3_sles_control)
#
# Usage:
#   bash launch_simulation_diffusion.sh [GOAL] [SEED] [INFERENCE_STEPS] [WARM_START]
#
#   GOAL             — optional, default "[-1.5, -1.5, 0, 0, 0]"
#   SEED             — optional random seed for obstacle placement (empty = random)
#   INFERENCE_STEPS  — optional DDIM denoising steps, default 5
#   WARM_START       — optional true/false, default true
#
# Example:
#   bash launch_simulation_diffusion.sh "[-1.5, -1.5, 0, 0, 0]" 42 5 true
# =============================================================================
set -e

GOAL="${1:-[-1.5, -1.5, 0, 0, 0]}"
SEED="${2:-}"
INFERENCE_STEPS="${3:-5}"
WARM_START="${4:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "${SCRIPT_DIR}")"

# Source environment
source "${WS_DIR}/sles_env.sh"

echo "=== SLES Simulation: Diffusion Policy Planner ==="
echo "    Goal             : ${GOAL}"
echo "    Seed             : ${SEED:-<random>}"
echo "    Inference steps  : ${INFERENCE_STEPS}"
echo "    Warm start       : ${WARM_START}"
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

# ── Step 3: Launch occupancy map node (background) ───────────────────────────
echo "[3/4] Launching LiDAR occupancy mapping node..."
ros2 launch turtlebot3_sles_perception turtlebot3_simulate_mapping.launch.py &
MAPPING_PID=$!
echo "      Mapping PID: ${MAPPING_PID}"

echo "      Waiting 3s for mapping node to initialize..."
sleep 3

# ── Step 4: Launch Diffusion planner (foreground) ─────────────────────────────
echo "[4/4] Launching Diffusion Policy planner..."
echo "      (Press Ctrl+C to stop all nodes)"
ros2 launch turtlebot3_sles_control turtlebot3_planner_diffusion.launch.py \
    goal:="${GOAL}" \
    inference_steps:="${INFERENCE_STEPS}" \
    use_warm_start:="${WARM_START}"

# Cleanup on exit
trap "kill ${GAZEBO_PID} ${MAPPING_PID} 2>/dev/null || true" EXIT
