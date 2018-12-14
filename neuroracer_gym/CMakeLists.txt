cmake_minimum_required(VERSION 2.8.3)
project(neuroracer_gym)

find_package(catkin REQUIRED COMPONENTS
  openai_ros
  rospy
)

catkin_python_setup()

catkin_package(

)

include_directories(
    ${catkin_INCLUDE_DIRS}
)