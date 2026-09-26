#!/usr/bin/env bash
set -euo pipefail
# Held by the supervisor, not inherited by long-lived children.
exec 9>/tmp/neuroracer-sim.lock
flock -n 9 || { echo 'A simulation is already running. Stop it before starting another; vnc can attach to it.' >&2; exit 1; }
child=
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$child" ]]; then
    kill -INT -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid ros2 launch neuroracer_sim sim.launch.py "web:=${1:-false}" 9>&- &
child=$!
wait "$child"
