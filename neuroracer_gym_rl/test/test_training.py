from functools import partial
import json
import os
import pickle
import time

import numpy as np
import pytest
import torch
import gymnasium as gym

import dqn
import training
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

    # A resumed run reopens the buffer with its data.
    reopened = ReplayBuffer((2, 2, 3), 12, str(tmp_path / 'buffer'))
    assert reopened.count[0] == 18 and np.array_equal(reopened.frames, buffer.frames)
    with pytest.raises(ValueError):
        ReplayBuffer((2, 2, 3), 24, str(tmp_path / 'buffer'))


def make_agent(name, working_dir):
    flipped = {'add_flipped': True} if name == 'dqn' else {}
    return __import__(name).Agent((64, 64, 2), 1 if name == 'ddpg' else 3, 64, working_dir=str(working_dir),
                                  batch_size=4, **flipped)


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
    batches = iter(loader(agent.buffer, agent.batch_size))
    agent.replay(next(batches))
    assert np.isfinite(agent.loss) and np.isfinite(agent.q)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    # The target moves once per epoch: 64 / 4 = 16 updates.
    assert all(torch.equal(old, new) for old, new in zip(ema_before, ema.module.parameters()))
    for _ in range(15):
        agent.replay(next(batches))
    ema_after = list(ema.module.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(ema_before, ema_after))
    assert any(not torch.equal(a, b) for a, b in zip(ema_after, model.parameters()))
    action = agent.act(rng.integers(0, 256, (1, 2, 64, 64), dtype=np.uint8))
    if name == 'ddpg':
        assert action.shape == (1,) and action.dtype == np.float32 and abs(action[0]) <= 1
    else:
        assert action in (0, 1, 2)
    agent.progress['steps'] = 24
    agent.save_model()
    assert agent.progress['updates'] == 16

    resumed = make_agent(name, tmp_path)
    resumed_model = resumed.actor if name == 'ddpg' else resumed.model
    resumed_ema = resumed.target_actor if name == 'ddpg' else resumed.target_model
    assert resumed.progress['steps'] == 24
    assert all(torch.equal(a, b) for a, b in zip(resumed_model.parameters(), model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(resumed_ema.module.parameters(), ema.module.parameters()))
    assert resumed.exploration_rate == agent.exploration_rate


def test_exploration_decays_linearly_with_collected_steps(tmp_path):
    agent = make_agent('dqn', tmp_path)
    for steps, rate in ((0, 1.0), (25000, 0.505), (50000, 0.01), (200000, 0.01)):
        agent.progress['steps'] = steps
        assert agent.exploration_rate == pytest.approx(rate)


def test_mirrored_transitions_swap_left_and_right(tmp_path):
    agent = make_agent('dqn', tmp_path)
    states = torch.arange(2 * 2 * 64 * 64).reshape(2, 2, 64, 64)
    batch = {'actions': torch.tensor([0, 2]), 'states': states, 'next_states': states + 1}
    flipped = agent.flip(batch, torch.tensor([True, False]))
    assert flipped['actions'].tolist() == [2, 2]
    assert torch.equal(flipped['states'][0], states[0].flip(-1)) and torch.equal(flipped['states'][1], states[1])
    assert torch.equal(flipped['next_states'][0], states[0].flip(-1) + 1)


class CameraEnv(gym.Env):
    def __init__(self, limit=5):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0, 255, (480, 640, 3), np.uint8)
        self.limit, self.count, self.closed = limit, 0, False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
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

small_dqn = partial(dqn.Agent, buffer_max_size=100, batch_size=4)


def test_run_saves_progress_and_resumes_to_the_total_steps(tmp_path):
    settings = dict(env_id='CameraEnv-v0', n_frames=2, sample_batch_size=10)
    game = NeuroRacer(small_dqn, str(tmp_path), n_steps=20, **settings)
    game.run()
    assert game.agent.progress['steps'] == 20 and game.agent.progress['episodes'] == 4
    assert game.env.unwrapped.closed
    episodes = [json.loads(line) for line in open(tmp_path / 'episodes.jsonl')]
    assert [e['steps'] for e in episodes] == [5] * 4 and all(e['crashed'] for e in episodes)

    game = NeuroRacer(small_dqn, str(tmp_path), n_steps=30, **settings)
    assert game.agent.progress['steps'] == 20
    assert game.agent.buffer.count[0] == 24  # 4 episodes of 5 transitions and a first frame
    game.run()
    assert game.agent.progress['steps'] == 30 and game.agent.buffer.count[0] == 36


def test_learner_trains_the_shared_agent_without_waiting(tmp_path):
    agent = make_agent('dqn', tmp_path)
    fill_buffer('dqn', agent.buffer)
    ema_before = [parameter.clone() for parameter in agent.target_model.module.parameters()]
    learner = Learner(agent, warmup_steps=7, metrics_path=str(tmp_path / 'updates.jsonl'))
    learner.start()
    time.sleep(15)
    learner.stop({'steps': 24, 'episodes': 3})
    assert learner.exitcode == 0
    checkpoint = load_checkpoint(agent.weight_backup)
    assert checkpoint['progress']['steps'] == 24 and checkpoint['progress']['updates'] > 16
    # The car's agent drives with the target the learner updated.
    assert any(not torch.equal(old, new) for old, new in zip(ema_before, agent.target_model.module.parameters()))
    assert all(torch.equal(saved.to(new.device), new) for saved, new in
               zip(checkpoint['target_model'].values(), agent.target_model.module.parameters()))
    (updates,) = [json.loads(line) for line in open(tmp_path / 'updates.jsonl')]
    assert updates['step'] == 24 and updates['updates'] == checkpoint['progress']['updates']
    assert np.isfinite(updates['loss'])


def write_config(tmp_path, text):
    (tmp_path / 'experiments').mkdir(exist_ok=True)
    path = tmp_path / 'experiments' / 'small.toml'
    path.write_text(text)
    return str(path)


def test_a_config_starts_one_run_that_resumes_only_with_the_same_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(NeuroRacer, 'run', lambda self: None)
    agent = '[agent]\nclass = "dqn.Agent"\nbuffer_max_size = 100\n'
    config = write_config(tmp_path, agent + '[training]\nenv_id = "CameraEnv-v0"\nn_steps = 100\n')
    training.main(['training.py', config])
    settings = json.load(open('runs/small/config.json'))
    assert settings['agent.class'] == 'dqn.Agent' and settings['agent.gamma'] == 0.99
    assert settings['training.n_steps'] == 100 and settings['training.max_episode_steps'] == 10000

    with pytest.raises(SystemExit, match='exists'):
        training.main(['training.py', config])
    write_config(tmp_path, agent + '[training]\nenv_id = "CameraEnv-v0"\nn_steps = 200\n')
    training.main(['training.py', config, '--resume'])
    assert json.load(open('runs/small/config.json'))['training.n_steps'] == 200
    write_config(tmp_path, agent + 'gamma = 0.9\n[training]\nenv_id = "CameraEnv-v0"\nn_steps = 200\n')
    with pytest.raises(SystemExit, match='agent.gamma'):
        training.main(['training.py', config, '--resume'])
    write_config(tmp_path, agent + 'gama = 0.9\n')
    with pytest.raises(TypeError, match='gama'):
        training.main(['training.py', config, '--resume'])
