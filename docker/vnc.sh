#!/usr/bin/env bash
set -euo pipefail
# A forcibly closed display should not leave large Qt/Fluxbox core dumps.
ulimit -c 0
exec 9>/tmp/neuroracer-vnc.lock
flock -n 9 || { echo 'VNC is already running on port 8091.' >&2; exit 1; }
# VNC is only a desktop client of the existing server, never a second server.
if flock -n /tmp/neuroracer-sim.lock true; then
  echo 'Start ./scripts/dev web (or sim) first.' >&2
  exit 1
fi
child=
watcher=
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$watcher" ]]; then kill -TERM "$watcher" 2>/dev/null || true; fi
  if [[ -n "$child" ]]; then
    kill -TERM -- "-$child" 2>/dev/null || true
    # Qt may linger after its X connection is closed. Bound shutdown for our
    # own process group, including grandchildren forked by gz / websockify.
    for _ in {1..30}; do
      kill -0 -- "-$child" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid xvfb-run -a -s '-screen 0 1280x800x24' bash docker/desktop.sh 9>&- &
child=$!
# A blocking lock watcher consumes no polling CPU and exits with the server.
flock /tmp/neuroracer-sim.lock true 9>&- &
watcher=$!
wait -n "$child" "$watcher"
