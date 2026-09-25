import math
import os
import pickle
import time

import numpy as np
import pytest
import torch
import gymnasium as gym

import utils
from utils import ReplayBuffer, ReplayDataset, ReplaySampler, loader, preprocess, load_checkpoint
from neuroracer_discrete import Learner, NeuroRacer

AGENTS = ('dqn', 'ddpg')


def test_preprocess_crops_scales_and_converts_to_grayscale():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:200] = 255  # cropped away
    image[200:, :, 0] = 100  # blue in BGR
    frame = preprocess(image, 200, 0.2, 0.2)
    assert frame.shape == (56, 128) and frame.dtype == np.uint8
    assert (frame == 11).all()  # 0.114 * 100


def test_buffer_rebuilds_frame_stacks_and_skips_overwritten_rows(tmp_path):
    buffer = ReplayBuffer((2, 2, 3), 12, str(tmp_path / 'buffer'))
    frame = lambda value: np.full((2, 2), value, np.uint8)
    expected, value = {}, 0
    for length in (2, 5, 1, 6):  # 18 rows wrap the 12-row buffer; row n holds value n.
        state = [value] * 3
        buffer.start_episode(frame(value))
        for step in range(length):
            value += 1
            next_state = state[1:] + [value]
            buffer.append(value % 3, frame(value), value, step == length - 1)
            expected[value] = (state, next_state, value % 3, step == length - 1)
            state = next_state
        value += 1
    # Another process maps the same files.
    dataset = ReplayDataset(pickle.loads(pickle.dumps(buffer)))
    seen = set()
    for n in expected:
        item = dataset[n]
        if item is None:
            continue
        states, next_states, action, terminate = expected[n]
        assert item['states'].shape == (3, 2, 2)
        assert item['states'][:, 0, 0].tolist() == states
        assert item['next_states'][:, 0, 0].tolist() == next_states
        assert (item['actions'].item(), item['rewards'].item(), item['terminates'].item()) == (action, n, terminate)
        seen.add(n)
    # Transitions whose earlier frames were overwritten are never returned or sampled.
    assert seen == {10, 12, 13, 14, 15, 16, 17}
    batches = iter(ReplaySampler(buffer, 64))
    assert all(set(next(batches)) <= seen for _ in range(10))
    buffer.close()
    assert not (tmp_path / 'buffer').exists()


def make_agent(name, working_dir):
    agent = __import__(name).Agent((64, 64, 2), 1 if name == 'ddpg' else 3, 64, add_flipped=name == 'dqn',
                                   working_dir=str(working_dir))
    agent.batch_size, agent.nb_steps_warmup = 4, 0
    return agent


def fill_buffer(name, buffer):
    rng = np.random.default_rng(0)
    for episode in range(3):
        buffer.start_episode(rng.integers(0, 256, (64, 64), dtype=np.uint8))
        for step in range(8):
            action = np.float32([rng.uniform(-1, 1)]) if name == 'ddpg' else int(rng.integers(3))
            buffer.append(action, rng.integers(0, 256, (64, 64), dtype=np.uint8), 1.0, step == 7)


@pytest.mark.parametrize('name', AGENTS)
def test_replay_trains_saves_and_resumes(name, tmp_path, monkeypatch):
    monkeypatch.setattr(utils, 'loader_workers', 0)
    agent = make_agent(name, tmp_path)
    fill_buffer(name, agent.buffer)
    rng = np.random.default_rng(0)
    model = agent.actor if name == 'ddpg' else agent.model
    ema = agent.target_actor if name == 'ddpg' else agent.target_model
    before = [parameter.detach().clone() for parameter in model.parameters()]
    ema_before = [parameter.detach().clone() for parameter in ema.module.parameters()]
    agent.replay(next(iter(loader(agent.buffer, agent.batch_size))))
    assert np.isfinite(agent.loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    ema_after = list(ema.module.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(ema_before, ema_after))
    assert any(not torch.equal(a, b) for a, b in zip(ema_after, model.parameters()))
    action = agent.act(rng.integers(0, 256, (1, 2, 64, 64), dtype=np.uint8))
    if name == 'ddpg':
        assert action.shape == (1,) and action.dtype == np.float32 and abs(action[0]) <= 1
    else:
        assert action in (0, 1, 2)
    agent.buffer.close()
    agent.progress['steps'] = 24
    agent.save_model()

    resumed = make_agent(name, tmp_path)
    resumed_model = resumed.actor if name == 'ddpg' else resumed.model
    resumed_ema = resumed.target_actor if name == 'ddpg' else resumed.target_model
    assert resumed.progress['steps'] == 24
    assert all(torch.equal(a, b) for a, b in zip(resumed_model.parameters(), model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(resumed_ema.module.parameters(), ema.module.parameters()))
    assert resumed.exploration_rate == agent.exploration_rate
    resumed.buffer.close()


def test_exploration_decays_linearly_with_collected_steps(tmp_path):
    agent = make_agent('dqn', tmp_path)
    for steps, rate in ((0, 1.0), (25000, 0.505), (50000, 0.01), (200000, 0.01)):
        agent.progress['steps'] = steps
        assert agent.exploration_rate == pytest.approx(rate)
    agent.buffer.close()


def test_mirrored_transitions_swap_left_and_right(tmp_path):
    agent = make_agent('dqn', tmp_path)
    states = torch.arange(2 * 2 * 64 * 64).reshape(2, 2, 64, 64)
    batch = {'actions': torch.tensor([0, 2]), 'states': states, 'next_states': states + 1}
    flipped = agent.flip(batch, torch.tensor([True, False]))
    assert flipped['actions'].tolist() == [2, 2]
    assert torch.equal(flipped['states'][0], states[0].flip(-1)) and torch.equal(flipped['states'][1], states[1])
    assert torch.equal(flipped['next_states'][0], states[0].flip(-1) + 1)
    agent.buffer.close()


class CameraEnv(gym.Env):
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
        time.sleep(0.02)
        self.count += 1
        return np.full(self.observation_space.shape, self.count, np.uint8), 1.0, self.count >= self.limit, False, {}

    def close(self):
        self.closed = True


gym.register('CameraEnv-v0', entry_point=CameraEnv)


def small_dqn(*args, **kwargs):
    agent = __import__('dqn').Agent(*args, **kwargs)
    agent.batch_size = 4
    return agent


def test_run_saves_progress_and_resumes(tmp_path):
    game = NeuroRacer(small_dqn, sample_batch_size=10, n_frames=2, buffer_max_size=100,
                      add_flipped=False, env_id='CameraEnv-v0', working_dir=str(tmp_path))
    game.run(20)
    assert game.agent.progress == {'steps': 20, 'episodes': 4}
    assert game.env.unwrapped.closed and not (tmp_path / 'buffer').exists()
    assert len(open(tmp_path / 'metrics.jsonl').readlines()) == 4

    game = NeuroRacer(small_dqn, sample_batch_size=10, n_frames=2, buffer_max_size=100,
                      add_flipped=False, env_id='CameraEnv-v0', working_dir=str(tmp_path))
    assert game.agent.progress == {'steps': 20, 'episodes': 4}
    assert game.agent.exploration_rate == pytest.approx(1 - 0.99 * 20 / 50000)
    game.agent.buffer.close()


def test_learner_trains_the_shared_agent_in_its_own_process(tmp_path):
    agent = make_agent('dqn', tmp_path)
    fill_buffer('dqn', agent.buffer)
    ema_before = [parameter.clone() for parameter in agent.target_model.module.parameters()]
    learner = Learner(agent)
    learner.start()
    deadline = time.time() + 60
    while math.isnan(learner.loss.value) and learner.exitcode is None and time.time() < deadline:
        time.sleep(0.1)
    learner.stop({'steps': 24, 'episodes': 3})
    assert learner.exitcode == 0 and np.isfinite(learner.loss.value)
    # The car's agent drives with the weights the learner updated.
    assert any(not torch.equal(old, new) for old, new in zip(ema_before, agent.target_model.module.parameters()))
    checkpoint = load_checkpoint(agent.weight_backup)
    assert checkpoint['progress'] == {'steps': 24, 'episodes': 3}
    assert all(torch.equal(saved.to(new.device), new) for saved, new in
               zip(checkpoint['target_model'].values(), agent.target_model.module.parameters()))
    agent.buffer.close()
