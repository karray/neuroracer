# Training

The runtime uses PyTorch 2.14.0 (CUDA 13.0 wheel) and Gymnasium 1.3.0. Start
`./scripts/dev web` or `sim` first and keep it running.

## Commands

```bash
./scripts/dev train dqn                          # runs/dqn, 200,000 steps
./scripts/dev train double_dqn --output runs/first-run --steps 50000
./scripts/dev train --help
```

`scripts/training.py` builds the `NeuroRacer` loop (`neuroracer_discrete.py`)
with the agent module of that name. If the output directory already holds the
agent's checkpoint (`<agent>_16f.pt`), it is loaded and training continues:
exploration then starts at its minimum, or at the saved rate with
`--always-explore true`. Ctrl-C stops training and saves the model.
`ros2 launch neuroracer_gym_rl start.launch agent:=dqn` starts the same, and
`ddpg_learning.launch` or `ddpg.py` trains DDPG on `NeuroRacer-v1`.
Evaluation is in `q_learning.ipynb` (`./scripts/dev notebook`).

## Environment

`NeuroRacer-v0` (`tasks/neuroracer_discrete_task.py`): actions 0/1/2 steer
right/straight/left (±1 rad) at `speed = 1`, which `_create_steering_command`
maps like racecar_control's `servo_commands.py` (about 0.5 m/s). The reward is
the lidar forward clearance minus the left/right imbalance; a collision ends the
episode with -100. `NeuroRacer-v1` steers continuously in [-1, 1] at
`speed = 10` with reward `1 - |action| - |steering|`.

- Observations are 480×640 BGR uint8 camera images.
- Each step advances exactly 0.1 s of physics (`multi_step` on the paused world)
  and returns the camera, lidar and odometry of that instant; the sensors run at
  10 Hz to match. The world is otherwise unthrottled, so steps run faster than
  real time. All waits have wall-clock timeouts.
- `reset` brakes the car and teleports it to `env.unwrapped.initial_position`
  (the spawn pose if unset) instead of resetting the world: Gazebo Jetty
  recreates plugins without a Reset hook on a world reset, which corrupts its
  heap under load. The training loop sets a random x in [1, 4] and a random
  heading for every episode. `info` holds the ground-truth `position` and `yaw`.
- Episodes have no time limit, as before; `gym.make(..., max_episode_steps=N)` adds one.
- ROS topics: `/camera/image_raw`, `/scan`, `/odom`, `/cmd_vel`
  (`angular.z` is yaw rate, not steering angle).

## Collection and training

The simulator is stepped on the main thread and every transition is appended to
the agent's cyclic replay buffer `H5Buffer` in `<output>/buffer.hdf5`. Its
capacity (1,000,000 transitions, about 7 GB) is limited by disk, not RAM: the
larger it is, the longer rare and old experience stays in training, which
counters catastrophic forgetting. Each preprocessed frame is stored once and
stacks are rebuilt when read. The file is deleted when training ends.

A learner thread calls the agent's `replay()` continuously as soon as the buffer
holds one batch. `replay()` works as before: it takes two chunks of 20,000
transitions (one while the buffer is smaller than two), now random contiguous
blocks read into GPU memory, computes their Q targets once with the model as
it is before the chunk (`double_*`: with the target model, which is updated
every 10 replays), and fits one shuffled epoch. The car then drives with the new
weights. The GPU runs targets and gradients in micro-batches of 128 (a batch's
gradients are summed before its optimizer step, so updates are unchanged) and
the learner waits for each one: CUDA executes work in the order it was queued,
so long or queued-up training work would delay every driving decision. Exploration decays every 1,000 collected steps and the model is saved
after the next replay.

## GPU

Training uses CUDA when PyTorch sees a GPU. For the container to see an NVIDIA
GPU, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host, then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
./scripts/dev up    # recreates the dev container with compose.gpu.yaml
```

`scripts/dev` adds `compose.gpu.yaml` whenever Docker has the NVIDIA runtime
(`NEURORACER_GPU=0` opts out); Gazebo then also renders sensors on the GPU.

## Agents

The agents are PyTorch ports of the original Keras models, with their
hyperparameters (Keras default initialization, Adam with lr 0.001, MSE, γ 0.9;
exploration 0.85, ×0.99 per 1,000 steps, down to 0.01):

| Agent | Model | Targets | Batch |
| --- | --- | --- | --- |
| dqn | CNN (16, 32, 64 filters, LeakyReLU, dropout) → 256 → 128 | the model | 1000 |
| double_dqn | Same CNN | target model | 1000 |
| drqn | Same CNN per frame → LSTM 512 | the model | 128 |
| double_drqn | Same CNN per frame → LSTM 512 | target model | 128 |
| ddpg | Actor and critic: 3 × Conv 32 → 200 (keras-rl DDPG) | soft target models (τ 0.001) | 16 |

Preprocessing crops the top 200 pixels, converts to grayscale, resizes to
56×128 and stacks 16 frames. DDPG explores with Ornstein-Uhlenbeck noise and
trains after 500 warmup steps. Only termination masks the Bellman bootstrap.

## Checkpoints and logs

`<agent>_16f.pt` is written atomically and holds the network and optimizer
states, the exploration rate and the step and episode counters. It is loaded
with `torch.load(weights_only=True)`. The replay buffer is not saved; after a
restart training continues once it holds one batch again.

`metrics.jsonl` records each episode's steps, return, exploration rate and the
last loss. Closing the environment stops the car and pauses the simulator.
