#!/usr/bin/env bash
set -e
mkdir -p "$HOME"
source /opt/ros/lyrical/setup.bash
if [ -f /workspace/.colcon/install/setup.bash ]; then
  source /workspace/.colcon/install/setup.bash
fi
exec "$@"
