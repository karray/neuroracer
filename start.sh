#!/bin/bash

HOME=/home
CATKIN_HOME=$HOME/catkin_ws
GZWEB_HOME=$HOME/gzweb

source "$CATKIN_HOME/devel/setup.bash"

nohup xvfb-run -s "-screen 0 640x480x24" roslaunch racecar_gazebo racecar_tunnel.launch &

cd $CATKIN_HOME/src/neuroracer
nohup jupyter lab --no-browser --LabApp.token='' --port 8888 &

cd $GZWEB_HOME
sleep 5 ; echo 'Strarting gzweb'; npm start