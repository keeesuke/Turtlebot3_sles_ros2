cmake_minimum_required(VERSION 3.8)
project(turtlebot3_sles_worlds)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch worlds models
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  spawn_random_world.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
