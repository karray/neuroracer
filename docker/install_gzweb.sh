#!/bin/bash

HOME=$1

MODEL_PATH="/home/catkin_ws/src/racecar-simulator/racecar_description/models"

apt-get install -yq --no-install-recommends libjansson-dev mercurial libboost-dev imagemagick libtinyxml-dev cmake build-essential

cd $HOME
hg clone https://bitbucket.org/osrf/gzweb
cd gzweb
hg up gzweb_1.4.0
# hg up default

source "/usr/share/gazebo/setup.sh"

# The first time you build, you'll need to gather all the Gazebo models which you want to simulate in the right directory ('http/client/assets') and prepare them for the web.
source /usr/share/gazebo/setup.sh
export GAZEBO_MODEL_PATH=$MODEL_PATH:$GAZEBO_MODEL_PATH
mkdir -p /home/gzweb/http/client/assets/
cp -rf /home/catkin_ws/src/racecar-simulator/racecar_description/ /home/gzweb/http/client/assets/

./deploy.sh -m local -t
# xvfb-run -s "-screen 0 640x480x24" ./deploy.sh -m local -t
