FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# FROM osrf/ros:melodic-desktop-full

# SHELL ["/bin/bash", "-c"]

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
# RUN mkdir ~/.gnupg && echo "disable-ipv6" >> ~/.gnupg/dirmngr.conf
# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# RUN apt-get install -y mercurial xvfb git libjansson-dev curl libboost-dev imagemagick libtinyxml-dev cmake build-essential
# RUN apt-get install -y ros-melodic-ackermann-msgs ros-melodic-effort-controllers ros-melodic-joy ros-melodic-tf2-sensor-msgs
 
# RUN mkdir -p /catkin_ws/src
# WORKDIR /catkin_ws/src

# RUN git clone https://github.com/mit-racecar/racecar.git
# RUN git clone https://github.com/mit-racecar/racecar-simulator.git

# WORKDIR /catkin_ws/
# RUN catkin_make

# RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
# RUN apt-get install -y nodejs && npm install -g npm

WORKDIR /home
COPY setup.sh start.sh ./
RUN ["chmod", "+x", "setup.sh"]
RUN ["chmod", "+x", "start.sh"]
RUN ./setup.sh

# RUN hg clone https://bitbucket.org/osrf/gzweb
# WORKDIR /gzweb
# RUN hg up gzweb_2.0.0

# RUN source /usr/share/gazebo/setup.sh && npm run deploy --- -m local && xvfb-run -s "-screen 0 640x480x24" npm run deploy --- -t


# RUN setup.sh

# Expose Jupyter 
EXPOSE 8888

# Expose Gazebo web 
EXPOSE 8080

# Expose Tensorboard
EXPOSE 6006

CMD ./start.sh