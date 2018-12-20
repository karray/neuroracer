cmake_minimum_required(VERSION 2.8.3)
project(neuroracer_gym_rl)

find_package(catkin REQUIRED COMPONENTS
  neuroracer_gym
  rospy
)


catkin_package(

)

include_directories(
    ${catkin_INCLUDE_DIRS}
)