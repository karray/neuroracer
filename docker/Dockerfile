FROM ros:lyrical-ros-base@sha256:0c19f326a339ed770ef1d4c0646a8b53bdb49dd5ff74b6de41ebdb8ac21e1806
# Upgrade keeps the base-image ROS libraries ABI-compatible with the newer ros_gz packages.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ros-lyrical-ros-gz ros-lyrical-cv-bridge python3-venv python3-numpy python3-opencv python3-matplotlib python3-pytest \
    xvfb xauth x11vnc novnc websockify fluxbox libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/neuroracer-venv
ENV PATH="/opt/neuroracer-venv/bin:${PATH}"
# The CUDA 13.0 wheel (NVIDIA driver 580 or newer) also runs without a GPU; build
# with --build-arg TORCH_VARIANT=cpu for a smaller CPU-only image.
ARG TORCH_VARIANT=cu130
RUN pip install --no-cache-dir torch==2.14.0+${TORCH_VARIANT} \
    --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# Gazebo's WebSocket plugin (the GzWeb server) from gz-sim10_10.5.0 with a disconnect fix,
# until https://github.com/gazebosim/gz-sim/pull/4016 is released.
ARG GZ_WEBSOCKET_SOURCE=https://raw.githubusercontent.com/gazebosim/gz-sim/2d3c7c777cc0a959666e3de497a38dca71b92409/src/systems/websocket_server
ADD --checksum=sha256:35cc30af20d991edfa0d7956cea0fec09a1c3fc396873976b2f7a311038b49f3 ${GZ_WEBSOCKET_SOURCE}/WebsocketServer.cc /tmp/websocket/
ADD --checksum=sha256:9c6e7444d9a95dccaf42d54cec96b78d966c78f81d65077daa56dbc14445c36f ${GZ_WEBSOCKET_SOURCE}/WebsocketServer.hh /tmp/websocket/
COPY websocket /tmp/websocket/
RUN cd /tmp/websocket && patch -p1 < disconnect.patch && . /opt/ros/lyrical/setup.sh \
    && cmake -S . -B build && cmake --build build -j && cmake --install build --prefix /opt/neuroracer \
    && rm -rf /tmp/websocket
