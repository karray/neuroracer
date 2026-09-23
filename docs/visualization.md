# Visualization

| Command | What runs | URL |
| --- | --- | --- |
| `./scripts/dev web` | Headless simulator, ROS bridge, GzWeb static server | http://localhost:8090/ |
| `./scripts/dev sim` | Headless simulator and ROS bridge only | — |
| `./scripts/dev vnc` | Gazebo desktop GUI attached to a running `web` or `sim` | http://localhost:8091/vnc.html?autoconnect=true&resize=scale |

Stop `web`/`sim` with Ctrl-C before switching modes; file locks reject a second
simulator or VNC session. `vnc` never starts another physics server and exits
when the simulator does. Both ports bind to host loopback only.

GzWeb opens in a third-person chase view of the car (**Follow car**; dragging
adjusts the offset, **Overview** stops following) with the car's own camera
sensor inset in the corner — the same image the agent receives. A paused world
produces no frames, so the inset updates only while the world runs (Play or
training); the **Camera** button toggles it. Leave Play/Pause alone while training, which owns
world timing. Neither viewer is part of the RL control path.

## Resource use

- One physics server and one ROS bridge serve all clients. Headless EGL renders
  only the robot's camera and lidar; no X server runs unless `vnc` is requested.
- The browser renders the scene at up to 30 FPS. The WebSocket plugin publishes
  at 20 Hz to at most four clients with 100 queued messages each. Hidden tabs
  disconnect and reconnect when visible.
- While the camera inset is shown, Gazebo PNG-encodes camera frames at up to
  20 Hz and keeps rendering the camera even when no ROS client listens. Turn the
  inset off to remove that cost.
- ROS bridges for camera, lidar, odometry and clock subscribe lazily. Mesa uses
  four worker threads (`LP_NUM_THREADS`); PyTorch uses `--threads`.
- Nginx serves the prebuilt frontend and proxies `/ws` to Gazebo's internal port
  9002, limited to 128 MiB and 32 processes.

## Maintenance

`docker/gzweb/` holds the frontend. `scripts/dev web` rebuilds it with Docker
cache; during a running session use
`docker compose --profile web up -d --build --no-deps web` and reload the page.

GzWeb 3.0.2 is pinned. Its npm release omits custom loaders, so the build fetches
the matching upstream commit by checksum. `patch-gzweb.mjs` applies small fixes
(message namespaces including camera images, disconnect cleanup, module imports,
Play request framing, initial and follow camera, render throttling) and fails the build if upstream source
changes. Review these patches when upgrading.

`ros2/neuroracer_websocket` builds Jetty 10.5.0's WebSocket plugin with one
disconnect accounting fix; without it a disconnect with queued messages leaves
its event thread spinning a full CPU core. Only `web` loads it; see
[UPSTREAM.md](../ros2/neuroracer_websocket/UPSTREAM.md). Remove it once upstream ships the fix.
