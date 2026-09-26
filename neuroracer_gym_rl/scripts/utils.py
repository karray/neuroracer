#!/usr/bin/env python3

import copy
from functools import partial
import math
import os

import cv2
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

loginfo = partial(print, flush=True)
context = mp.get_context('spawn')
# Processes that read replay batches from disk.
loader_workers = 4


def preprocess(img, y_offset, x_scale, y_scale, interpolation=cv2.INTER_AREA):
    return cv2.resize(cv2.cvtColor(img[y_offset:,:], cv2.COLOR_BGR2GRAY), None, fx=x_scale, fy=y_scale, interpolation=interpolation)


class ReplayBuffer():
    # A row that starts an episode holds its first frame; every other row holds the frame
    # after a transition, with that transition's action, reward and termination.
    # `count` is the number of rows written so far; row n is stored at n % maxlen.
    # An existing buffer in `path` is reopened, so a resumed run continues with its data.
    def __init__(self, state_shape, maxlen, path='buffer', action_shape=(), action_dtype=np.ubyte):
        self.maxlen = maxlen
        self.n_frames = state_shape[2]
        self.path = path
        os.makedirs(path, exist_ok=True)

        def array(name, shape, dtype):
            file = os.path.join(path, name + '.npy')
            if not os.path.exists(file):
                return np.lib.format.open_memmap(file, mode='w+', dtype=dtype, shape=shape)
            existing = np.load(file, mmap_mode='r+')
            if existing.shape != shape or existing.dtype != dtype:
                raise ValueError(file + ' does not match the buffer size, frame size or action type')
            return existing
        self.frames = array('frames', (maxlen,) + state_shape[:2], np.uint8)
        self.first = array('first', (maxlen,), np.bool_)
        self.actions = array('actions', (maxlen,) + action_shape, action_dtype)
        self.rewards = array('rewards', (maxlen,), np.float32)
        self.terminates = array('terminates', (maxlen,), np.bool_)
        self.count = array('count', (1,), np.int64)

    def __getstate__(self):
        # Other processes map the same files instead of receiving a copy of them.
        return {'maxlen': self.maxlen, 'n_frames': self.n_frames, 'path': self.path}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name in ('frames', 'first', 'actions', 'rewards', 'terminates', 'count'):
            setattr(self, name, np.load(os.path.join(self.path, name + '.npy'), mmap_mode='r'))

    def _write(self, frame, first, action=0, reward=0.0, terminate=False):
        # The row is complete before the count includes it.
        idx = int(self.count[0]) % self.maxlen
        self.frames[idx], self.first[idx], self.actions[idx] = frame, first, action
        self.rewards[idx], self.terminates[idx] = reward, terminate
        self.count[0] += 1

    def start_episode(self, frame):
        self._write(frame, True)

    def append(self, action, next_frame, reward, terminate):
        self._write(next_frame, False, action, float(reward), bool(terminate))


class ReplayDataset(Dataset):
    """Transition n of the buffer, or None if the buffer overwrote it while it was read."""
    def __init__(self, buffer):
        self.buffer = buffer

    def __getitem__(self, n):
        buffer, maxlen = self.buffer, self.buffer.maxlen
        rows = np.arange(n - buffer.n_frames, n + 1)
        # Stacks repeat an episode's first frame rather than reach into the previous episode.
        rows = np.maximum(rows, rows[buffer.first[rows % maxlen]].max(initial=rows[0]))
        # The state and the next state share all but one image, so each image travels once.
        item = {'actions': torch.tensor(buffer.actions[n % maxlen]),
                'frames': torch.from_numpy(buffer.frames[rows % maxlen]),
                'rewards': torch.tensor(buffer.rewards[n % maxlen]),
                'terminates': torch.tensor(buffer.terminates[n % maxlen])}
        # Row r is overwritten while row r + maxlen is written.
        if rows[0] <= buffer.count[0] - maxlen:
            return None
        return item


class ReplaySampler(Sampler):
    """Endless batches of uniformly random transitions, without the oldest 1% of the
    buffer, which is overwritten next."""
    def __init__(self, buffer, batch_size):
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self):
        buffer = self.buffer
        while True:
            count = int(buffer.count[0])
            low = max(count - buffer.maxlen + buffer.n_frames + buffer.maxlen // 100, 1)
            rows = np.empty(0, np.int64)
            while len(rows) < self.batch_size:
                candidates = np.random.randint(low, count, self.batch_size)
                rows = np.concatenate((rows, candidates[~buffer.first[candidates % buffer.maxlen]]))
            yield rows[:self.batch_size].tolist()


def collate(items):
    return default_collate([item for item in items if item is not None])


def loader(buffer, batch_size):
    return DataLoader(ReplayDataset(buffer), batch_sampler=ReplaySampler(buffer, batch_size),
                      num_workers=loader_workers, collate_fn=collate, pin_memory=torch.cuda.is_available(),
                      multiprocessing_context='spawn' if loader_workers else None,
                      persistent_workers=loader_workers > 0)


def to_device(batch, device):
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    frames = batch.pop('frames')
    return {**batch, 'states': frames[:, :-1], 'next_states': frames[:, 1:]}


def convnet(state_size, outputs):
    height, width, frames = state_size
    return nn.Sequential(
        nn.Conv2d(frames, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * math.ceil(height / 16) * math.ceil(width / 16), outputs),
    )


class Normalize(nn.Module):
    def forward(self, states):
        return states.float() / 255.0


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def fit(model, batch, loss):
    model.optimizer.zero_grad()
    batch_loss = loss(batch)
    batch_loss.backward()
    model.optimizer.step()
    return float(batch_loss.detach())


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class EMA():
    # The learner process updates the weights in place while the car's process drives with them.
    # GPU work is finished before the lock is released, and the learner holds it only for the update.
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.lock = context.Lock()

    def update(self, model):
        device = next(model.parameters()).device
        synchronize(device)
        with self.lock, torch.no_grad():
            for average, parameter in zip(self.module.parameters(), model.parameters()):
                average.lerp_(parameter, 1 - self.decay)
            synchronize(device)

    def __call__(self, *inputs):
        with torch.inference_mode(), self.lock, autocast(inputs[0].device):
            outputs = self.module(*inputs).float()
            synchronize(inputs[0].device)
        return outputs


def save_checkpoint(path, **state):
    # Write then rename, so Ctrl-C never leaves a truncated checkpoint.
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)
