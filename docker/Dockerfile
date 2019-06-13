FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

WORKDIR /home
COPY setup.sh start.sh install_ros.sh clone_build.sh install_gzweb.sh requirements.txt ./
RUN ["chmod", "+x", "setup.sh", "start.sh", "install_ros.sh", "clone_build.sh", "install_gzweb.sh"]
RUN ./setup.sh

# Expose Jupyter 
EXPOSE 8888
# Expose Gazebo web 
EXPOSE 8080

CMD ./start.sh