# Visualization

| Command | What runs | URL |
| --- | --- | --- |
| `./scripts/dev web` | Headless simulator, ROS bridge, GzWeb static server | http://localhost:8090/ |
| `./scripts/dev sim` | Headless simulator and ROS bridge only | — |
| `./scripts/dev vnc` | Gazebo desktop GUI attached to a running `web` or `sim` | http://localhost:8091/vnc.html?autoconnect=true&resize=scale |

Stop `web`/`sim` with Ctrl-C before switching modes. Both ports bind to host
loopback only.

GzWeb follows the car (**Follow car**, **Overview**) and shows its camera image
in the corner (**Camera**). Leave Play/Pause alone while training.

## Maintenance

`docker/gzweb/` holds the frontend. `scripts/dev web` rebuilds it with Docker
cache; during a running session use
`docker compose --profile web up -d --build --no-deps web` and reload the page.

GzWeb 3.0.2 is pinned and patched by `patch-gzweb.mjs`, which fails the build if
the upstream source changes.

`neuroracer_websocket` is Jetty 10.5.0's WebSocket plugin with a disconnect fix
(see [UPSTREAM.md](../neuroracer_websocket/UPSTREAM.md)).
