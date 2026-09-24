import os
import time

import numpy as np
import pytest
import torch
import gymnasium as gym

from utils import H5Buffer, preprocess
from neuroracer_discrete import NeuroRacer

AGENTS = ('dqn', 'double_dqn', 'drqn', 'double_drqn', 'ddpg')


def test_preprocess_crops_and_scales_to_uint8():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:200] = 255
    frame = preprocess(image, 200, 0.2, 0.2)
    assert frame.shape == (56, 128) and frame.dtype == np.uint8 and frame.max() == 0


def test_buffer_blocks_rebuild_frame_stacks(tmp_path):
    buffer = H5Buffer((2, 2, 3), 12, str(tmp_path / 'buffer.hdf5'))
    frame = lambda value: np.full((2, 2), value, np.uint8)
    expected, value = {}, 0
    for length in (2, 5, 1, 6):  # 18 rows wrap the 12-row buffer.
        state = [value] * 3
        buffer.start_episode(frame(value))
        for step in range(length):
            value += 1
            next_state = state[1:] + [value]
            buffer.append(value % 3, frame(value), value, step == length - 1)
            expected[value] = (state, next_state, value % 3, step == length - 1)
            state = next_state
        value += 1
    seen = set()
    for _ in range(50):
        block = buffer.sample(4)
        batch = block.batch(block.transitions)
        for index, reward in enumerate(batch['rewards'].tolist()):
            states, next_states, action, terminate = expected[reward]
            assert batch['states'][index][:, 0, 0].tolist() == states
            assert batch['next_states'][index][:, 0, 0].tolist() == next_states
            assert (batch['actions'][index].item(), batch['terminates'][index].item()) == (action, terminate)
            seen.add(reward)
    # Transitions whose earlier frames were overwritten are never sampled.
    assert seen == {10, 12, 13, 14, 15, 16, 17}
    buffer.close()
    assert not (tmp_path / 'buffer.hdf5').exists()


def make_agent(name, working_dir, **kwargs):
    module = __import__(name)
    action_size = 1 if name == 'ddpg' else 3
    agent = module.Agent((12, 16, 2), action_size, 64, 16, add_flipped=name == 'dqn', working_dir=str(working_dir), **kwargs)
    agent.batch_size, agent.nb_steps_warmup = 4, 0
    return agent


@pytest.mark.parametrize('name', AGENTS)
def test_replay_trains_saves_and_resumes(name, tmp_path):
    agent = make_agent(name, tmp_path)
    rng = np.random.default_rng(0)
    for episode in range(3):
        agent.buffer.start_episode(rng.integers(0, 256, (12, 16), dtype=np.uint8))
        for step in range(8):
            action = np.float32([rng.uniform(-1, 1)]) if name == 'ddpg' else int(rng.integers(3))
            agent.buffer.append(action, rng.integers(0, 256, (12, 16), dtype=np.uint8), 1.0, step == 7)
    model = agent.actor if name == 'ddpg' else agent.model
    before = [parameter.detach().clone() for parameter in model.parameters()]
    agent.progress['steps'] = 24
    agent.save_requested = True
    agent.replay()
    assert np.isfinite(agent.loss) and not agent.save_requested
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    # The acting copy is refreshed after each replay.
    assert all(torch.equal(a, b) for a, b in zip(agent.policy.copy.parameters(), model.parameters()))
    action = agent.act(rng.integers(0, 256, (1, 2, 12, 16), dtype=np.uint8))
    if name == 'ddpg':
        assert action.shape == (1,) and action.dtype == np.float32 and abs(action[0]) <= 1
    else:
        assert action in (0, 1, 2)
    agent.buffer.close()

    resumed = make_agent(name, tmp_path)
    resumed_model = resumed.actor if name == 'ddpg' else resumed.model
    assert resumed.progress['steps'] == 24
    assert all(torch.equal(a, b) for a, b in zip(resumed_model.parameters(), model.parameters()))
    if name != 'ddpg':
        assert resumed.exploration_rate == resumed.exploration_min
    resumed.buffer.close()


class CameraEnv(gym.Env):
    """Deterministic image environment for the collection loop without ROS."""
    def __init__(self, limit=5):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0, 255, (480, 640, 3), np.uint8)
        self.limit, self.count, self.closed = limit, 0, False
        self.initial_position = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        assert 1 <= self.initial_position['p_x'] <= 4
        self.count = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        assert self.action_space.contains(action)
        time.sleep(0.02)  # A simulator step takes tens of milliseconds.
        self.count += 1
        return np.full(self.observation_space.shape, self.count, np.uint8), 1.0, self.count >= self.limit, False, {}

    def close(self):
        self.closed = True


gym.register('CameraEnv-v0', entry_point=CameraEnv)


def small_dqn(*args, **kwargs):
    agent = __import__('dqn').Agent(*args, **kwargs)
    agent.batch_size = 4
    return agent


def test_learner_trains_while_collecting_and_run_resumes(tmp_path):
    game = NeuroRacer(small_dqn, sample_batch_size=10, n_frames=2, buffer_max_size=100, chunk_size=16,
                      add_flipped=False, env_id='CameraEnv-v0', working_dir=str(tmp_path))
    game.run(20)
    agent = game.agent
    assert agent.progress == {'steps': 20, 'episodes': 4}
    assert agent.loss is not None  # Trained from the first batch, beside collection.
    assert agent.exploration_rate == pytest.approx(0.85 * 0.99 ** 2)
    assert game.env.unwrapped.closed and not (tmp_path / 'buffer.hdf5').exists()
    assert len(open(tmp_path / 'metrics.jsonl').readlines()) == 4

    game = NeuroRacer(small_dqn, sample_batch_size=10, n_frames=2, buffer_max_size=100, chunk_size=16,
                      add_flipped=False, env_id='CameraEnv-v0', working_dir=str(tmp_path))
    assert game.agent.progress == {'steps': 20, 'episodes': 4}
    assert game.agent.exploration_rate == game.agent.exploration_min
    game.agent.buffer.close()
