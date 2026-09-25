# Training

Start `./scripts/dev web` or `sim` first and keep it running.

```bash
./scripts/dev train dqn                          # runs/dqn, 200,000 steps
./scripts/dev train dqn --output runs/first-run --steps 50000
./scripts/dev train dqn --frames 4                # stack 4 camera images per state
./scripts/dev train --help
```

An existing checkpoint (`<output>/<agent>_<frames>f.pt`) is loaded and training
continues. Ctrl-C stops training and saves the model. `ddpg.py` or
`ddpg_learning.launch` trains DDPG on `NeuroRacer-v1`. Evaluation is in
`q_learning.ipynb` (`./scripts/dev notebook`).

## Environment

- `NeuroRacer-v0`: actions 0/1/2 steer right/straight/left at 0.5 m/s.
  `NeuroRacer-v1`: continuous steering in [-1, 1].
- Observations are 480×640 BGR camera images.
- Each step advances the paused world by 0.1 s.
- Each episode starts at a random x in [1, 4] with a random heading. Training
  truncates episodes after 1,200 steps (`--max-episode-steps`).

## Training

Training runs in two processes that share the agent:

- The car's process steps the simulator, picks actions with the EMA of the
  network's weights, writes each transition to the replay buffer and logs every
  episode to `metrics.jsonl` in the output directory.
- The learner process (`Learner` in `neuroracer_discrete.py`) trains the network
  continuously on uniformly random batches that four `DataLoader` worker
  processes read from the buffer. It updates the EMA, which is also the target
  network, after each step and saves the checkpoint every 1,000 collected steps
  and when training ends.

They share:

- The replay buffer: memory-mapped files in `<output>/buffer/` (1,000,000
  transitions, about 7 GB), deleted when training ends. The car writes and the
  workers read. A transition that was overwritten while it was read is dropped
  from its batch.
- The agent's tensors, through CUDA IPC. The learner updates them in place and a
  lock keeps the car from driving with a half-updated EMA.
- The collected steps and episodes for the checkpoint, the latest loss, and the
  save and stop signals.

Side effects:

- Both processes run at their own speed, so the number of updates per collected
  step depends on the hardware. On an RTX 3060 Ti the car collects about 22
  steps/s and the learner makes about 1.7 updates/s, so each transition is
  sampled about 20 times on average.
- The car drives with weights at most one update old, and a run is not
  reproducible from `--seed`.
- Ctrl-C stops the car's process, which lets the learner finish its update and
  save.
- The loss in `metrics.jsonl` is `NaN` until the first update.

| Agent | Network | EMA decay | Batch | Loss |
| --- | --- | --- | --- | --- |
| dqn | convolutions → 512 → 3 Q-values | 0.995 | 256 | Huber, Double DQN, γ 0.99 |
| ddpg | convolutions → actor and critic | 0.999 | 16 | MSE, γ 0.9 |

The networks start with four 3×3 convolutions with stride 2 (32, 64, 64 and 128
channels) on 56×128 grayscale images: the camera image without its top 200
rows, scaled by 0.2. DQN explores with ε falling linearly from 1.0 to 0.01 over
50,000 steps, DDPG with Ornstein-Uhlenbeck noise.

## GPU

Training uses CUDA when the container has an NVIDIA GPU. Install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host, then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
./scripts/dev up
```

`scripts/dev` adds `compose.gpu.yaml` when Docker has the NVIDIA runtime
(`NEURORACER_GPU=0` opts out). The two training processes share GPU memory
through CUDA IPC, which WSL 2 does not support.
