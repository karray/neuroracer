# Training

Start `./scripts/dev web` or `sim` first and keep it running.

```bash
./scripts/dev train experiments/dqn.toml            # a new run in runs/dqn
./scripts/dev train experiments/dqn.toml --resume   # continue it
```

An experiment is a TOML file in `experiments/`:

- `[agent]` names the agent class and overrides defaults of its keyword
  arguments, e.g. `gamma`, `learning_rate`, `batch_size`, `ema_decay`, the
  exploration schedule and `buffer_max_size` of `dqn.Agent`.
- `[training]` overrides defaults of `NeuroRacer`'s keyword arguments: the task
  `env_id`, `n_epochs` in total (250), `max_episode_steps` (10,000),
  `n_frames`, `warmup_steps` and the save interval `sample_batch_size`.
- `seed` seeds Python, NumPy and PyTorch.

A different network is a new agent class, and a different reward a new task
registered with gymnasium; a config then names it.

The run folder `runs/<config name>/` holds `config.json` with every value used,
the checkpoint, `episodes.jsonl`, `updates.jsonl` and the replay buffer. A new run
refuses an existing folder. `--resume` continues a run when the config differs
only in `n_epochs`, with its replay buffer. Ctrl-C stops training and saves the
model. `ddpg.py` or `ddpg_learning.launch` trains `experiments/ddpg.toml` (DDPG on
`NeuroRacer-v1`). Evaluation is in `q_learning.ipynb` (`./scripts/dev notebook`).

## Environment

- `NeuroRacer-v0`: actions 0/1/2 steer right/straight/left at 0.5 m/s.
  `NeuroRacer-v1`: continuous steering in [-1, 1].
- Observations are 480×640 BGR camera images.
- Each step advances the paused world by 0.1 s.
- Each episode starts at one of four points (x = 1, 2, 3, 4 at y = 3.7) with a
  random heading. Training truncates episodes after `max_episode_steps`.

## Training

Training runs in two processes that share the agent:

- The car's process steps the simulator, picks actions with the EMA of the
  network's weights, writes each transition to the replay buffer and logs every
  episode to `episodes.jsonl`: length, return, whether it crashed, start point.
- The learner process (`Learner` in `neuroracer_discrete.py`) starts after
  `warmup_steps` collected steps and then trains the network continuously on
  uniformly random batches that four `DataLoader` worker processes read from
  the buffer. Once per epoch, as many updates as a full buffer has batches
  (1,000,000 / 256 ≈ 3,900 for DQN), it moves the EMA, which is also the target
  network, towards the network. Training ends after `n_epochs` epochs, so every
  config learns from the same number of sampled transitions (an epoch is
  `buffer_max_size` of them, whatever the batch size); the car collects until
  then. Every 1,000 collected steps and when training ends it saves the
  checkpoint and appends to `updates.jsonl`: the number of updates, mean loss
  and Q-value, and dropped transitions.

They share:

- The replay buffer: memory-mapped files in `runs/<config name>/buffer/`
  (1,000,000 transitions, about 7 GB), kept for `--resume`; delete runs you no
  longer need. The car writes and the workers read. A transition that was
  overwritten while it was read is dropped from its batch.
- The agent's tensors, through CUDA IPC. The learner updates them in place and a
  lock keeps the car from driving with a half-updated EMA.
- The collected steps and episodes for the checkpoint, and the save and stop
  signals.

Side effects:

- Neither process waits for the other, so the number of updates per collected
  step, and with it the steps a run collects, depends on the hardware and the
  config: on an RTX 3060 Ti about 2.5 for one frame (81 updates/s at about 30
  steps/s), about 1.1 for 16 frames (reading the frame stacks limits the
  learner to about 34 updates/s). `updates.jsonl` records it.
- The car drives with the EMA of the latest epoch, and a run is not
  reproducible from `seed`.
- Ctrl-C stops the car's process, which lets the learner finish its update and
  save.

| Agent | Network | EMA decay per update | Batch | Loss |
| --- | --- | --- | --- | --- |
| dqn | convolutions → 512 → 3 Q-values | 0.9998 | 256 | Huber, Double DQN, γ 0.99 |
| ddpg | convolutions → actor and critic | 0.999 | 16 | MSE, γ 0.9 |

`ema_decay` is per update, as in timm's `ModelEmaV3`, and compounds over the
epoch: DQN's EMA moves 1 − 0.9998^3,906 ≈ 54% of the way to the network once per
epoch, DDPG's (0.999 over 62,500 updates) practically the whole way.

The networks start with four 3×3 convolutions with stride 2 (32, 64, 64 and 128
channels) on 56×128 grayscale images: the camera image without its top 200
rows, scaled by 0.2. `n_frames` stacks the last camera images as input channels
(`experiments/dqn-16f.toml`: 16, so the network sees motion; an episode's first
image repeats until there are enough). DQN explores with ε falling linearly
from 1.0 to 0.01 over 50,000 steps, DDPG with Ornstein-Uhlenbeck noise.

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
