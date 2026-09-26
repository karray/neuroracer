# Visualization

| Command | What runs | URL |
| --- | --- | --- |
| `./scripts/dev web` | Headless simulator, ROS bridge, GzWeb static server | http://localhost:8090/ |
| `./scripts/dev sim` | Headless simulator and ROS bridge only | — |
| `./scripts/dev vnc` | Gazebo desktop GUI attached to a running `web` or `sim` | http://localhost:8091/vnc.html?autoconnect=true&resize=scale |

Stop `web`/`sim` with Ctrl-C before switching modes. Both ports bind to host
loopback only.

GzWeb only shows the simulation: it follows the car (**Follow car**,
**Overview**), shows its camera image in the corner (**Camera**) and what the
training processes publish (**Telemetry**: the run, `train` or `drive`, and the
car's and the learner's progress). The world stays paused except while
`./scripts/dev train` or `drive` steps it. Clicking a telemetry field switches
it between its value and a graph of its last 300 values (reward, action, loss,
Q and updates/s start as graphs); a section disappears 5 s after its process
stops publishing.

## Maintenance

`docker/gzweb/` holds the frontend. `scripts/dev web` rebuilds it with Docker
cache; during a running session use
`docker compose --profile web up -d --build --no-deps web` and reload the page.

GzWeb 3.0.2 is pinned and patched by `patch-gzweb.mjs`, which fails the build if
the upstream source changes.

The image builds Jetty 10.5.0's WebSocket plugin from pinned upstream source
with [a disconnect fix](../docker/websocket/disconnect.patch). Drop it once
[gz-sim#4016](https://github.com/gazebosim/gz-sim/pull/4016) is released.
