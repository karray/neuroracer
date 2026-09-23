#!/usr/bin/env bash
set -e
mkdir -p "$HOME"
source /opt/ros/lyrical/setup.bash
if [ -f /workspace/.ros2/install/setup.bash ]; then
  source /workspace/.ros2/install/setup.bash
fi
exec "$@"
