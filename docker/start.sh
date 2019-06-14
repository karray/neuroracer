#!/bin/bash

HOME=/home
CATKIN_HOME=$HOME/catkin_ws
GZWEB_HOME=$HOME/gzweb

source "$CATKIN_HOME/devel/setup.bash"

nohup xvfb-run -s "-screen 0 640x480x24" roslaunch racecar_gazebo racecar_tunnel.launch &

nohup jupyter lab --notebook-dir $HOME --port 8888 --ip 0.0.0.0 --allow-root --no-browser --LabApp.token='' &

cd $GZWEB_HOME
sleep 10; echo 'Strarting gzweb'; npm start