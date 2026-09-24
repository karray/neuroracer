import json
from pathlib import Path
import threading
import time
import gymnasium as gym
import numpy as np
from .agents import make_agent
from .checkpoint import save_checkpoint, load_checkpoint
from .replay import FrameHistory, ReplayBuffer


def make_environment(config, max_episode_steps):
    import neuroracer_gym  # Register ROS 2 environments only when using the simulator.
    name = 'NeuroRacerContinuous-v0' if config.continuous else 'NeuroRacerDiscrete-v0'
    return gym.make(name, max_episode_steps=max_episode_steps)


class Learner(threading.Thread):
    """Trains while the caller collects. Each epoch reads a random replay block, updates
    on its transitions in shuffled batches, then refreshes the policy the agent acts with."""
    def __init__(self, agent, replay, config):
        super().__init__(name='learner', daemon=True)
        self.agent, self.replay, self.config = agent, replay, config
        self.rng = np.random.default_rng(config.seed)
        self.lock = threading.Lock()  # Held for each update, so checkpoints see a consistent agent.
        self.stopping = threading.Event()
        self.metrics, self.error = {}, None

    def run(self):
        batch_size = self.config.batch_size
        try:
            while not self.stopping.is_set():
                block = (self.replay.block(self.config.block_size, self.agent.device)
                         if len(self.replay) > batch_size else None)
                if block is None or len(block.transitions) < batch_size:
                    self.stopping.wait(0.1)
                    continue
                rows = self.rng.permutation(block.transitions)
                for start in range(0, len(rows) - batch_size + 1, batch_size):
                    if self.stopping.is_set():
                        return
                    with self.lock:
                        self.metrics = self.agent.update(block.batch(rows[start:start + batch_size]))
                self.agent.sync_policy()
        except BaseException as error:
            self.error = error

    def stop(self):
        self.stopping.set()
        if self.ident is not None:
            self.join()


def train(steps, output, *, config=None, resume=None, checkpoint_interval=1000,
          log_interval=100, max_episode_steps=1000, env_factory=make_environment):
    """Train for `steps` more transitions from a new `config` or a `resume` checkpoint.

    A resumed run keeps the checkpoint's configuration and refills a fresh replay buffer.
    The simulator is stepped on the calling thread while a `Learner` trains beside it.
    """
    output = Path(output)
    checkpoint = output / 'latest.pt'
    if resume:
        agent, step, episodes = load_checkpoint(resume)
        config = agent.config
    elif checkpoint.exists():
        raise FileExistsError('Checkpoint already exists; use --resume or a new --output directory')
    else:
        agent, step, episodes = make_agent(config), 0, 0
    output.mkdir(parents=True, exist_ok=True)
    replay = ReplayBuffer(output / 'replay.h5', config, config.seed)
    history = FrameHistory(config)
    learner = Learner(agent, replay, config)
    started = time.monotonic()
    interrupted = False
    env = None
    with (output / 'metrics.jsonl').open('a') as log:
        def save():
            with learner.lock:
                save_checkpoint(checkpoint, agent, step, episodes)

        def record(event):
            event = {'step': step, 'episodes': episodes, 'elapsed_seconds': time.monotonic() - started, **event}
            log.write(json.dumps(event, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(event), flush=True)
        try:
            env = env_factory(config, max_episode_steps)
            learner.start()
            image, _ = env.reset(seed=config.seed + episodes)
            state = history.reset(image)
            replay.start_episode(state[-1])
            episode_return = 0.0
            for _ in range(steps):
                if learner.error is not None:
                    raise RuntimeError('Learner failed') from learner.error
                action = agent.act(state, step=step, explore=True)
                image, reward, terminated, truncated, _ = env.step(action)
                next_state = history.append(image)
                # Keep the terminal observation; a reset observation starts a new episode.
                replay.append(action, reward, next_state[-1], terminated)
                state = next_state
                step += 1
                episode_return += float(reward)
                if terminated or truncated:
                    episodes += 1
                    record({'event': 'episode', 'return': episode_return,
                            'terminated': bool(terminated), 'truncated': bool(truncated)})
                    image, _ = env.reset(seed=config.seed + episodes)
                    state = history.reset(image)
                    replay.start_episode(state[-1])
                    episode_return = 0.0
                if step % log_interval == 0:
                    record({'event': 'train', 'replay_size': len(replay), **learner.metrics,
                            **({} if config.continuous else {'epsilon': config.epsilon(step)})})
                if step % checkpoint_interval == 0:
                    save()
        except KeyboardInterrupt:
            interrupted = True
            record({'event': 'interrupted'})
        finally:
            # Save learned progress even if a later ROS call fails or Ctrl-C stops training.
            try:
                learner.stop()
                save()
            finally:
                replay.close()
                if env is not None:
                    env.close()
        result = {'event': 'complete', 'checkpoint': str(checkpoint), 'updates': agent.updates,
                  'interrupted': interrupted, **learner.metrics}
        record(result)
    return {'step': step, 'episodes': episodes, **result}


def evaluate(checkpoint, *, episodes=3, max_episode_steps=1000, device='cpu',
             env_factory=make_environment):
    agent, _, _ = load_checkpoint(checkpoint, device=device)
    history = FrameHistory(agent.config)
    results = []
    with env_factory(agent.config, max_episode_steps) as env:
        for episode in range(episodes):
            image, _ = env.reset(seed=agent.config.seed + episode)
            state = history.reset(image)
            steps, episode_return, done = 0, 0.0, False
            while not done:
                image, reward, terminated, truncated, _ = env.step(agent.act(state, explore=False))
                state = history.append(image)
                steps += 1
                episode_return += float(reward)
                done = terminated or truncated
            result = {'episode': episode + 1, 'steps': steps, 'return': episode_return}
            results.append(result)
            print(json.dumps(result), flush=True)
    return results
