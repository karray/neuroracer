![](assets/neuroracer.gif)

# NeuroRacer

A camera-based reinforcement-learning racecar simulator built on ROS 2 Lyrical,
Gazebo Jetty, Gymnasium 1.3 and PyTorch 2.14, running in Docker.

## Setup

Requires Docker Engine with Compose v2 on Linux amd64 and about 15 GB of free
disk space. Rendering is CPU/software by default.

```bash
./scripts/dev setup      # build image, start container, colcon build, unit tests
./scripts/dev web        # headless Gazebo + browser viewer at http://localhost:8090/
```

Keep `web` running. The world starts paused; training controls pause/advance
itself. Use `./scripts/dev sim` instead of `web` for training without a browser,
and `./scripts/dev vnc` for the full Gazebo desktop (see [visualization](docs/visualization.md)).
Only one simulator and one controller may run at a time.

```bash
./scripts/dev check-sim  # drive/steer/reset integration check (simulator required)
./scripts/dev check      # unit tests (no simulator required)
./scripts/dev shell      # development shell
./scripts/dev build      # rebuild after adding package files or dependencies
./scripts/dev down       # stop everything
```

Source is bind-mounted and colcon uses symlink installation, so Python edits
apply on process restart. Build artifacts go to the ignored `.ros2/` directory.
The container runs with your UID/GID; save data under `/workspace`.

## Gymnasium environment

```python
import gymnasium as gym
import neuroracer_gym

with gym.make('NeuroRacerDiscrete-v0') as env:
    image, info = env.reset()
    image, reward, terminated, truncated, info = env.step(1)  # drive straight
```

- `NeuroRacerDiscrete-v0`: actions 0/1/2 steer right/straight/left.
  `NeuroRacerContinuous-v0`: a float32 array with one steering value in [-1, 1].
- Observations are 480×640 RGB uint8 camera images. Steering is limited to
  ±0.6 rad at a constant 1 m/s.
- Each step advances about 0.1 s of simulation and waits for fresh sensor data;
  the world is paused between steps. All waits have wall-clock timeouts.
- Reward is lidar forward clearance minus left/right imbalance; a collision
  terminates the episode with -100. Episodes truncate after 1000 steps
  (`gym.make(..., max_episode_steps=N)` overrides this).
- `reset(options={'pose': (x, y, yaw)})` sets the start pose (default `(2, 3.7, pi/2)`).
- ROS topics: `/camera/image_raw`, `/scan`, `/odom`, `/clock`, `/cmd_vel`
  (`angular.z` is yaw rate, not steering angle).

## Training

With `web` or `sim` running, in a second terminal:

```bash
./scripts/dev train double_dqn --steps 100000 --output runs/first-run
./scripts/dev train --resume runs/first-run/latest.pt --steps 10000
./scripts/dev eval runs/first-run/latest.pt --episodes 3
```

Agents: `dqn`, `double_dqn`, `drqn`, `double_drqn` (discrete) and `ddpg`
(continuous). See [training details](docs/pytorch-training.md).
`q_learning.ipynb` shows the same Python API.
