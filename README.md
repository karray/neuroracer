# Software requirements #
* Ubuntu 18.04 or Windows WSL Ubuntu 18.04
* Python 2.7
* Tenserflow CPU or GPU

# Installation #
### ROS Melodic for Ubuntu 18.04 ###
```bash
sudo apt install ros-melodic-desktop-full

sudo apt install ros-melodic-ackermann-msgs
sudo apt install ros-melodic-effort-controllers
sudo apt install ros-melodic-joy
sudo apt install ros-melodic-tf2-sensor-msgs
```

### Creating catkin workspace ###
```bash
cd ~
mkdir catkin_ws
cd catkin_ws/
mkdir src
cd src/
```

### MIT Racecar ###
```bash
cd ~/catkin_ws/src/
git clone https://github.com/mit-racecar/racecar.git
git clone https://github.com/mit-racecar/racecar-simulator.git
```

### openai_ros ###
```bash
git clone https://bitbucket.org/theconstructcore/openai_ros.git
```

### Environment ###
```bash
git clone https://github.com/karray/neuroracer.git
````
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash 
```

sudo pip install tensorflow gym keras

roslaunch racecar_gazebo racecar_tunnel.launch
roslaunch neuroracer_gym_rl qlearning.launch

# neuroracer_gym
xvfb-run -s "-screen 0 1280x1024x24" npm run deploy --- -t

windows xserver for camera

process has died exit code -9: The script needed too much memory

laser bug.

sumulation start delay
