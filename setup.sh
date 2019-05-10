HOME=/home

CATKIN_HOME=$HOME/catkin_ws
CATKIN_SRC=$CATKIN_HOME/src

mkdir -p $CATKIN_SRC

apt-get update && \
    apt-get install -y git sb-release lsb-core curl wget xvfb \
    python-pip python-dev #libgtk2.0-0 unzip libblas-dev liblapack-dev libhdf5-dev

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
# apt-get update
# apt-get install ros-melodic-desktop-full
# rosdep init
# rosdep update
# echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# source /opt/ros/melodic/setup.bash

apt-get install -y ros-melodic-ackermann-msgs ros-melodic-effort-controllers os-melodic-joy ros-melodic-tf2-sensor-msgs

cd $CATKIN_SRC
git clone https://github.com/mit-racecar/racecar.git
git clone https://github.com/mit-racecar/racecar-simulator.git

git clone https://bitbucket.org/theconstructcore/openai_ros.git

git clone https://github.com/karray/neuroracer.git

cd $CATKIN_HOME
catkin_make

# Gazebo Web
apt-get install -y libjansson-dev mercurial libboost-dev imagemagick libtinyxml-dev cmake build-essential

curl -sL https://deb.nodesource.com/setup_10.x | bash -
apt-get install -y nodejs && npm install -g npm

cd $HOME
git clone https://bitbucket.org/osrf/gzweb
cd gzweb
git checkout default

source /usr/share/gazebo/setup.sh

xvfb-run -s "-screen 0 640x480x24" npm run ./deploy.sh -m local -t

# Python libs
pip install numpy scipy matplotlib scikit-learn cv2 gym keras

pip install tensorflow-gpu