cmake_minimum_required(VERSION 3.20)
project(neuroracer_gym_rl)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/training.py
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES
  scripts/neuroracer_discrete.py
  scripts/utils.py
  scripts/dqn.py
  scripts/ddpg.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
