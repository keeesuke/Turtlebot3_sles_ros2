cmake_minimum_required(VERSION 3.8)
project(turtlebot3_sles_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts as executables
install(PROGRAMS
  turtlebot3_sles_perception/simulate_lidar_publisher.py
  turtlebot3_sles_perception/simulate_lidar_publisher_new.py
  turtlebot3_sles_perception/sim_mapping.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
