sudo apt install ros-melodic-desktop-full
sudo apt install ros-melodic-joy
sudo apt install ros-melodic-tf2-sensor-msgs

cd ~/catkin_ws/src/
git clone https://github.com/mit-racecar/racecar-simulator.git
git clone https://bitbucket.org/theconstructcore/openai_ros.git
git clone https://github.com/karray/neuroracer.git

cd ~/catkin_ws
catkin_make
source devel/setup.bash 

sudo pip install tensorflow gym keras

roslaunch neuroracer_gym_rl qlearning.launch

# neuroracer_gym
xvfb-run -s "-screen 0 1280x1024x24" npm run deploy --- -t

windows xserver for camera

process has died exit code -9: The script needed too much memory

laser bug.

sumulation start delay
