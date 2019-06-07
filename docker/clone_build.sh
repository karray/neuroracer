#!/bin/bash

CATKIN_HOME=$1/catkin_ws
CATKIN_SRC=$CATKIN_HOME/src

mkdir -p $CATKIN_SRC

cd $CATKIN_SRC
git clone https://github.com/mit-racecar/racecar.git
git clone https://github.com/mit-racecar/racecar-simulator.git

git clone https://bitbucket.org/theconstructcore/openai_ros.git

git clone https://github.com/karray/neuroracer.git

cd $CATKIN_HOME
source "/opt/ros/melodic/setup.bash" && catkin_make