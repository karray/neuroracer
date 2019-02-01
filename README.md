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
[openai_ros](http://wiki.ros.org/openai_ros) package provides OpenAI Gym environments which allows to compare Reinforcement Learning algorithms by providing a common API.
```bash
git clone https://bitbucket.org/theconstructcore/openai_ros.git
```

### Environment ###
This project implements openai_ros API for MIT Racear robot.
```bash
git clone https://github.com/karray/neuroracer.git
````
```bash
cd ~/catkin_ws
catkin_make
```

The following python packages are required
```bash
sudo pip install tensorflow gym keras
````


# Usage #
After this MIT ROS-package and this project have to be started in their own terminals.
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch racecar_gazebo racecar_tunnel.launch
```

```bash
source ~/catkin_ws/devel/setup.bash 
roslaunch neuroracer_gym_rl qlearning.launch
```
# WSL and headless setup #
Sometimes headless set up is needed. For example, when there is only ssh access to the server or if the server runs on Windows.

Headless set up has a couple of special requirements. In order to get the camera rendering a view,  you will need an xserver running. This can be achieved in several ways. The universal solution is Xvfb.

### Xvfb ###
>Xvfb or X virtual framebuffer is a display server implementing the X11 display server protocol. In contrast to other display servers, Xvfb performs all graphical operations in virtual memory without showing any screen output
>
>https://en.wikipedia.org/wiki/Xvfb

First install xvfb
```bash
sudo apt install xvfb
```
Then start the project
```bash

```

### Gzweb ###
Gzweb can be otionally installed
>Gzweb is a WebGL client for Gazebo. Like gzclient, it's a front-end graphical interface to gzserver and provides visualization of the simulation. However, Gzweb is a thin client in comparison, and lets you interact with the simulation from the comfort of a web browser. This means cross-platform support, minimal client-side installation, and support for mobile devices.
>
>http://gazebosim.org/gzweb.html

```bash
xvfb-run -s "-screen 0 640x480x24" npm run deploy --- -t
```

# neuroracer_gym


windows xserver for camera

process has died exit code -9: The script needed too much memory

laser bug.

sumulation start delay
