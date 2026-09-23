# Training

The runtime uses PyTorch 2.14.0 (CPU wheel) and Gymnasium 1.3.0. Start
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

Hyperparameters (`--frames`, `--batch-size`, `--buffer-size`, `--warmup`,
`--learning-rate`, `--gamma`, `--target-interval`, `--epsilon-steps`, `--seed`,
`--threads`, DDPG's `--actor-learning-rate`, `--tau`, `--noise-std`, …) mirror
the fields of `neuroracer_rl.config.Config`. On `--resume` the checkpoint's
settings are used and overrides are rejected.

The container is CPU-only. GPU training (`--device cuda`) requires a CUDA
PyTorch wheel and Docker GPU configuration.

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
bootstraps from the final observation. Replay stores read-only uint8 frames
shared between adjacent transitions (roughly 70 MiB for 10,000 transitions).

## Checkpoints and logs

`latest.pt` is written atomically every `--checkpoint-interval` steps and when
training finishes, is interrupted with Ctrl-C, or fails. It holds the
configuration, network and optimizer states, counters and RNG states, and is
loaded with `torch.load(weights_only=True)`. The replay buffer is not saved;
after resume it refills to `warmup` before updates continue.

`metrics.jsonl` records losses, exploration rate, episode returns and elapsed
time. Closing the environment stops the car and pauses the simulator.
