#!/bin/bash

apt-get install --no-install-recommends -yq lsb-release

sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

apt-get update

apt-get install -yq --no-install-recommends ros-melodic-desktop-full \
    ros-melodic-ackermann-msgs ros-melodic-effort-controllers ros-melodic-joy ros-melodic-tf2-sensor-msgs
    
rosdep init
rosdep update
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
