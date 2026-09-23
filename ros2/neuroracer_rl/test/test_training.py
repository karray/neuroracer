import numpy as np
import pytest
import torch
import gymnasium as gym
from neuroracer_rl.config import Config, ALGORITHMS
from neuroracer_rl.replay import FrameHistory, ReplayBuffer
from neuroracer_rl.agents import make_agent, bootstrap_targets, q_next_values
from neuroracer_rl.checkpoint import save_checkpoint, load_checkpoint
from neuroracer_rl.runner import train, evaluate


def small_config(algorithm='double_dqn'):
    return Config(algorithm=algorithm, frames=2, batch_size=2, buffer_size=8,
                  warmup=2, target_interval=1, tau=1.0, threads=1)


def make_batch(config):
    rng = np.random.default_rng(2)
    shape = (config.batch_size, config.frames, config.height, config.width)
    return {'states': rng.integers(0, 256, shape, dtype=np.uint8),
            'next_states': rng.integers(0, 256, shape, dtype=np.uint8),
            'actions': (np.array([[0.1], [-0.2]], dtype=np.float32) if config.continuous
                        else np.array([0, 2])),
            'rewards': np.array([1.0, -1.0], dtype=np.float32),
            'terminated': np.array([True, False])}


def test_truncation_bootstraps_but_termination_does_not():
    result = bootstrap_targets(torch.tensor([1., 1.]), torch.tensor([True, False]),
                               torch.tensor([10., 10.]), 0.9)
    torch.testing.assert_close(result, torch.tensor([1., 10.]))


def test_double_dqn_selects_online_action_and_evaluates_target():
    online = torch.tensor([[10., 0., 1.]])
    target = torch.tensor([[1., 5., 2.]])
    assert q_next_values(online, target, double=True).item() == 1
    assert q_next_values(online, target, double=False).item() == 5


def test_frame_history_crops_rgb_and_shares_immutable_frames():
    history = FrameHistory(small_config())
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:200] = 255
    image[200:, :, 0] = 255
    first = history.reset(image)
    assert first[0].shape == (56, 128) and first[0].dtype == np.uint8
    assert first[0].mean() == 76  # RGB red, not BGR blue, and sky crop removed.
    second = history.append(np.zeros_like(image))
    assert second[0] is first[1]
    assert not second[0].flags.writeable
    assert first[-1].mean() == 76  # Appending cannot mutate an old replay state.
    reset = history.reset(np.ones_like(image) * 255)
    assert all(frame.mean() == 255 for frame in reset)


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_learning_update_and_checkpoint_roundtrip(algorithm, tmp_path):
    config = small_config(algorithm)
    agent = make_agent(config)
    model = agent.actor if config.continuous else agent.online
    before = [param.detach().clone() for param in model.parameters()]
    metrics = agent.update(make_batch(config))
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    target = agent.actor_target if config.continuous else agent.target
    assert all(torch.equal(a, b) for a, b in zip(target.parameters(), model.parameters()))
    state = tuple(make_batch(config)['states'][0])
    predicted = agent.act(state, explore=False)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, agent, 17, 3)
    restored, step, episodes = load_checkpoint(path)
    assert (step, episodes, restored.updates) == (17, 3, 1)
    np.testing.assert_array_equal(restored.act(state, explore=False), predicted)
    # Optimizer and target states must resume the exact next numerical update.
    agent.update(make_batch(config))
    restored.update(make_batch(config))
    restored_model = restored.actor if config.continuous else restored.online
    for left, right in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(left, right)
    if config.continuous:
        assert predicted.shape == (1,) and predicted.dtype == np.float32 and abs(predicted[0]) <= 1
    else:
        assert predicted in (0, 1, 2)


class CameraEnv(gym.Env):
    """Small deterministic image environment for testing the full runner without ROS."""
    def __init__(self, config, limit):
        self.action_space = (gym.spaces.Box(-1, 1, (1,), np.float32) if config.continuous
                             else gym.spaces.Discrete(3))
        self.observation_space = gym.spaces.Box(0, 255, (480, 640, 3), np.uint8)
        self.limit = limit
        self.count = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.count += 1
        return np.full(self.observation_space.shape, self.count, dtype=np.uint8), 1.0, False, self.count >= self.limit, {}

    def close(self):
        self.closed = True


@pytest.mark.parametrize('algorithm', ('double_dqn', 'ddpg'))
def test_train_resume_and_inference_do_not_overwrite_checkpoint(algorithm, tmp_path):
    config = small_config(algorithm)
    result = train(4, tmp_path, config=config, max_episode_steps=2, env_factory=CameraEnv)
    assert result['step'] == 4 and result['updates'] == 3 and result['episodes'] == 2
    path = tmp_path / 'latest.pt'
    resumed = train(2, tmp_path, resume=path, max_episode_steps=2, env_factory=CameraEnv)
    assert resumed['step'] == 6 and resumed['updates'] == 4
    before = path.read_bytes()
    episodes = evaluate(path, episodes=2, max_episode_steps=2, env_factory=CameraEnv)
    assert episodes == [{'episode': 1, 'steps': 2, 'return': 2.0}, {'episode': 2, 'steps': 2, 'return': 2.0}]
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        train(1, tmp_path, config=config, env_factory=CameraEnv)


@pytest.mark.parametrize('failure', (KeyboardInterrupt, RuntimeError))
def test_interruption_and_ros_failure_save_progress_and_close(failure, tmp_path):
    instances = []

    class FailingEnv(CameraEnv):
        def __init__(self, config, limit):
            super().__init__(config, limit)
            instances.append(self)

        def step(self, action):
            if self.count == 1:
                raise failure('simulated interruption')
            return super().step(action)

    if failure is KeyboardInterrupt:
        result = train(3, tmp_path, config=small_config(), env_factory=FailingEnv)
        assert result['interrupted']
    else:
        with pytest.raises(RuntimeError, match='simulated interruption'):
            train(3, tmp_path, config=small_config(), env_factory=FailingEnv)
    _, step, _ = load_checkpoint(tmp_path / 'latest.pt')
    assert step == 1 and instances[0].closed

