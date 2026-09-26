#!/usr/bin/env bash
set -eo pipefail
children=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${children[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Keep Gazebo inside the virtual desktop, including when Qt restores old geometry.
mkdir -p "$HOME/.fluxbox"
cat > "$HOME/.fluxbox/apps" <<'APPS'
[app] (title=.*Gazebo.*)
  [Position] (UPPERLEFT) {0 0}
  [Maximized] {yes}
[end]
APPS
fluxbox > /tmp/neuroracer-fluxbox.log 2>&1 &
children+=("$!")
x11vnc -display "$DISPLAY" -forever -shared -nopw -listen 127.0.0.1 -rfbport 5900 > /tmp/neuroracer-vnc.log 2>&1 &
children+=("$!")
websockify --web=/usr/share/novnc 0.0.0.0:8091 127.0.0.1:5900 &
children+=("$!")
export GZ_SIM_RESOURCE_PATH="/workspace/.colcon/install/neuroracer_sim/share/neuroracer_sim/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim -g &
children+=("$!")
echo 'Gazebo Jetty: http://localhost:8091/vnc.html?autoconnect=true&resize=scale'
wait -n
