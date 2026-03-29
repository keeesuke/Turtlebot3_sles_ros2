cmake_minimum_required(VERSION 3.8)
project(turtlebot3_sles_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts as executables
install(PROGRAMS
  turtlebot3_sles_control/planner_haa_only.py
  turtlebot3_sles_control/planner_haa_real_world.py
  turtlebot3_sles_control/planner_nn.py
  turtlebot3_sles_control/planner_diffusion.py
  turtlebot3_sles_control/planner_switch_mpc_nn.py
  turtlebot3_sles_control/map_occupancy_plotter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
