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

Transitions go to a replay buffer on disk (`<output>/buffer.hdf5`, 1,000,000
transitions, up to 150 GB), which is deleted when training ends. A learner
thread trains continuously on random contiguous chunks of 5,000 transitions,
one shuffled epoch per chunk, while the car collects. The car drives with the
EMA of the network's weights, which is also the target network.

| Agent | Network | EMA decay | Batch | Loss |
| --- | --- | --- | --- | --- |
| dqn | resnet18 → 3 Q-values | 0.995 | 256 | Huber, Double DQN, γ 0.9 |
| ddpg | resnet18 actor and critic | 0.999 | 16 | MSE, γ 0.9 |

The networks are timm `resnet18` with GroupNorm, trained from scratch on 224×224
RGB images (the camera image without its top 200 rows), 3 channels per frame.
DQN explores with ε falling linearly from 1.0 to 0.01 over 50,000 steps, DDPG
with Ornstein-Uhlenbeck noise.

`metrics.jsonl` in the output directory records every episode.

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
(`NEURORACER_GPU=0` opts out).
