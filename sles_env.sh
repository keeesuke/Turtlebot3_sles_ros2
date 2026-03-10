#!/usr/bin/env bash
# sles_env.sh — Environment setup for TurtleBot3 SLES ROS2 Humble workspace
#
# Usage (inside Docker or on host after sourcing ROS2):
#   source /illinois/Spring2025/independent_study/turtlebot3_ws/sles_env.sh
#
# This script:
#   1. Sources ROS2 Humble global setup
#   2. Sources the colcon workspace overlay
#   3. Sets TURTLEBOT3_MODEL
#   4. Sets GAZEBO_MODEL_PATH to include sles_worlds models
#   5. Optionally sets LDS_MODEL and RMW implementation

# ── 1. ROS2 Humble base ──────────────────────────────────────────────────────
if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo "[sles_env] WARNING: /opt/ros/humble/setup.bash not found." >&2
fi

# ── 2. Workspace overlay ─────────────────────────────────────────────────────
SLES_WS="/illinois/Spring2025/independent_study/turtlebot3_ws"

# Prefer local install directory overlay; fall back to install/setup.bash
if [ -f "${SLES_WS}/install/setup.bash" ]; then
    # shellcheck disable=SC1090
    source "${SLES_WS}/install/setup.bash"
else
    echo "[sles_env] NOTE: Workspace overlay not found — run colcon build first." >&2
fi

# ── 3. TurtleBot3 model ──────────────────────────────────────────────────────
# Options: burger | waffle | waffle_pi
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"

# ── 4. LDS sensor model ──────────────────────────────────────────────────────
export LDS_MODEL="${LDS_MODEL:-LDS-01}"

# ── 5. RMW implementation (Fast-DDS is ROS2 Humble default) ─────────────────
# export RMW_IMPLEMENTATION=rmw_fastrtps_cpp   # default — uncomment to be explicit
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp # alternative

# ── 6. Gazebo model search path ──────────────────────────────────────────────
SLES_WORLDS_SHARE="${SLES_WS}/install/turtlebot3_sles_worlds/share/turtlebot3_sles_worlds"
if [ -d "${SLES_WORLDS_SHARE}/models" ]; then
    export GAZEBO_MODEL_PATH="${SLES_WORLDS_SHARE}/models:${GAZEBO_MODEL_PATH:-}"
fi

# Also add official turtlebot3_gazebo models if available
TB3_SIM_SHARE="${SLES_WS}/install/turtlebot3_gazebo/share/turtlebot3_gazebo"
if [ -d "${TB3_SIM_SHARE}/models" ]; then
    export GAZEBO_MODEL_PATH="${TB3_SIM_SHARE}/models:${GAZEBO_MODEL_PATH:-}"
fi

# ── 7. Summary ───────────────────────────────────────────────────────────────
echo "[sles_env] TurtleBot3 SLES environment loaded."
echo "           TURTLEBOT3_MODEL = ${TURTLEBOT3_MODEL}"
echo "           LDS_MODEL        = ${LDS_MODEL}"
echo "           GAZEBO_MODEL_PATH= ${GAZEBO_MODEL_PATH}"
