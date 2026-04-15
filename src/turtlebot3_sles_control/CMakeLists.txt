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
  turtlebot3_sles_control/planner_nn_real_world.py
  turtlebot3_sles_control/planner_diffusion.py
  turtlebot3_sles_control/planner_switch_mpc_nn.py
  turtlebot3_sles_control/planner_switch_mpc_nn_real_world.py
  turtlebot3_sles_control/map_occupancy_plotter.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install trained model weights alongside the scripts so __file__-relative
# path resolution works in both symlink-install and copy-install builds.
install(FILES
  turtlebot3_sles_control/best_model.pth
  turtlebot3_sles_control/best_model_diff.pth
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
