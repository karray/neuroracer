#!/bin/bash

HOME=/home
CATKIN_HOME=$HOME/catkin_ws
GZWEB_HOME=$HOME/gzweb

source "$CATKIN_HOME/devel/setup.bash"

nohup xvfb-run -s "-screen 0 640x480x24" roslaunch racecar_gazebo racecar_tunnel.launch &
cd $GZWEB_HOME
echo 'Strarting gzweb'
sleep 10 ; npm start