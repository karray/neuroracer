#!/bin/bash

HOME=/home

CATKIN_HOME=$HOME/catkin_ws
CATKIN_SRC=$CATKIN_HOME/src

mkdir -p $CATKIN_SRC

echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# export DEBIAN_FRONTEND="noninteractive"

# apt-get update && apt-get install -yq --no-install-recommends apt-utils

apt-get update && apt-get install --no-install-recommends -yq git lsb-core curl wget xvfb \
    python-pip python-dev \
    dirmngr gnupg2 lsb-release
    #libgtk2.0-0 unzip libblas-dev liblapack-dev libhdf5-dev

sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116

apt-get update

apt-get install -yq --no-install-recommends ros-melodic-desktop-full
rosdep init
rosdep update
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source /opt/ros/melodic/setup.bash

# Setup ros env
# /ros_entrypoint.sh


apt-get install -yq ros-melodic-ackermann-msgs ros-melodic-effort-controllers ros-melodic-joy ros-melodic-tf2-sensor-msgs

# apt-get install -yq sb-release


cd $CATKIN_SRC
git clone https://github.com/mit-racecar/racecar.git
git clone https://github.com/mit-racecar/racecar-simulator.git

git clone https://bitbucket.org/theconstructcore/openai_ros.git

git clone https://github.com/karray/neuroracer.git

cd $CATKIN_HOME
source "/opt/ros/melodic/setup.bash" && catkin_make

# Gazebo Web
apt-get install -yq --no-install-recommends libjansson-dev mercurial libboost-dev imagemagick libtinyxml-dev cmake build-essential

curl -sL https://deb.nodesource.com/setup_10.x | bash -
apt-get install -y nodejs && npm install -g npm
rm -rf /var/lib/apt/lists/*

cd $HOME
hg clone https://bitbucket.org/osrf/gzweb
cd gzweb
hg up default

source "/usr/share/gazebo/setup.sh"

# xvfb-run -s "-screen 0 640x480x24" ./deploy.sh -m local -t
./deploy.sh -m local -t

# Python libs
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

pip install numpy scipy matplotlib scikit-learn opencv-python gym keras

pip install jupyterlab

pip install tensorflow-gpu==1.13.1