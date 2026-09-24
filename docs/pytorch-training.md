# Training

The runtime uses PyTorch 2.14.0 (CUDA 13.0 wheel) and Gymnasium 1.3.0. Start
`./scripts/dev web` or `sim` first and keep it running.

## Commands

```bash
./scripts/dev train double_dqn --steps 100000 --output runs/first-run
./scripts/dev train --resume runs/first-run/latest.pt --steps 10000
./scripts/dev eval runs/first-run/latest.pt --episodes 3
./scripts/dev train --help
```

Without `--output`, a timestamped directory is created under `runs/`. A fresh
run never overwrites an existing checkpoint. `--steps` on resume means
additional steps. `eval` runs without exploration, learning or checkpoint writes.

Hyperparameters (`--frames`, `--batch-size`, `--buffer-size`, `--block-size`, `--warmup`,
`--learning-rate`, `--gamma`, `--target-interval`, `--epsilon-steps`, `--seed`,
`--threads`, DDPG's `--actor-learning-rate`, `--tau`, `--noise-std`, …) mirror
the fields of `neuroracer_rl.config.Config`. On `--resume` the checkpoint's
settings are used and overrides are rejected.

## Collection and training

The simulator is stepped on the main thread and every transition is appended to
a cyclic replay buffer in `<output>/replay.h5`. Its capacity (`--buffer-size`,
default 1,000,000, about 7 GB) is limited by disk, not RAM: the larger it is,
the longer rare and old experience stays in training, which counters
catastrophic forgetting. Each frame is stored once and stacks are rebuilt when
read. The file is deleted when training ends.

A learner thread starts as soon as the buffer holds one batch and trains
continuously: each epoch reads a random contiguous block (`--block-size`,
default 10,000 transitions; HDF5 reads blocks much faster than scattered rows)
into GPU memory, updates on it in shuffled batches (`--batch-size`, default
128) whose frame stacks are gathered on the GPU, and then copies the weights to
the policy network the car drives with. Thread timing varies, so runs with the
same seed are not bit-identical. With an RTX 3060 Ti the learner keeps the GPU
about 90% busy (about 150 updates/s) while the simulator runs at about 20 steps/s.

## GPU

`--device auto` (the default) trains on CUDA when PyTorch sees a GPU. For the
container to see an NVIDIA GPU, install the
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

| Agent | Model | Learning rule |
| --- | --- | --- |
| dqn | CNN over stacked grayscale frames | Target-network DQN |
| double_dqn | Same CNN | Online network selects, target network evaluates |
| drqn | Per-frame CNN followed by an LSTM | Recurrent DQN |
| double_drqn | Same CNN/LSTM | Recurrent Double DQN |
| ddpg | Separate CNN actor and critic | Deterministic actor/critic, soft target updates |

Preprocessing crops the top 200 pixels, converts to grayscale, resizes to
56×128 and stacks 16 frames. Recurrent agents encode each frame and run the LSTM
across the window, so no hidden state is carried between replay samples. DQN
variants use Huber loss; all agents clip gradients. DDPG explores with clipped
Gaussian noise after a uniform-random warmup.

Only termination masks the Bellman bootstrap; a time-limit truncation still
bootstraps from the final observation.

## Checkpoints and logs

`latest.pt` is written atomically every `--checkpoint-interval` steps and when
training finishes, is interrupted with Ctrl-C, or fails. It holds the
configuration, network and optimizer states, counters and RNG states, and is
loaded with `torch.load(weights_only=True)`. The replay buffer is not saved;
after resume training continues once it holds one batch again.

`metrics.jsonl` records losses, exploration rate, episode returns and elapsed
time. Closing the environment stops the car and pauses the simulator.
