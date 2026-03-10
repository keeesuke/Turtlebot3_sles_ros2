cmake_minimum_required(VERSION 3.8)
project(turtlebot3_sles_data)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  turtlebot3_sles_data/robot_data_recorder_mpc.py
  turtlebot3_sles_data/analyze_robot_data.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
