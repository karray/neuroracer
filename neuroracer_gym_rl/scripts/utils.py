#!/usr/bin/env python3

from functools import partial
import os
import shutil

import cv2
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate
from timm.utils import ModelEmaV3

loginfo = partial(print, flush=True)
context = mp.get_context('spawn')
# Samples per forward and backward pass; a whole batch of 256 needs about 7 GB of GPU memory.
micro_batch_size = 64
# Processes that read replay batches from disk.
loader_workers = 4


def preprocess(img, y_offset, size, interpolation=cv2.INTER_AREA):
    img = cv2.resize(img[y_offset:,:], (size, size), interpolation=interpolation)
    return np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))


class ReplayBuffer():
    # A row that starts an episode holds its first frame; every other row holds the frame
    # after a transition, with that transition's action, reward and termination.
    # `count` is the number of rows written so far; row n is stored at n % maxlen.
    def __init__(self, state_shape, maxlen, path='buffer', action_shape=(), action_dtype=np.ubyte):
        self.maxlen = maxlen
        self.n_frames = state_shape[2]
        self.path = path
        os.makedirs(path, exist_ok=True)

        def array(name, shape, dtype):
            return np.lib.format.open_memmap(os.path.join(path, name + '.npy'), mode='w+', dtype=dtype, shape=shape)
        self.frames = array('frames', (maxlen, 3) + state_shape[:2], np.uint8)
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

    def length(self):
        return min(int(self.count[0]), self.maxlen)

    def close(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)


class ReplayDataset(Dataset):
    """Transition n of the buffer, or None if the buffer overwrote it while it was read."""
    def __init__(self, buffer):
        self.buffer = buffer

    def __getitem__(self, n):
        buffer, maxlen = self.buffer, self.buffer.maxlen
        rows = np.arange(n - buffer.n_frames, n + 1)
        # Stacks repeat an episode's first frame rather than reach into the previous episode.
        rows = np.maximum(rows, rows[buffer.first[rows % maxlen]].max(initial=rows[0]))
        frames = buffer.frames[rows % maxlen]
        item = {'actions': torch.tensor(buffer.actions[n % maxlen]),
                'states': torch.from_numpy(frames[:-1]).flatten(0, 1),
                'next_states': torch.from_numpy(frames[1:]).flatten(0, 1),
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
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class Normalize(nn.Module):
    def forward(self, states):
        return states.float() / 255.0


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def fit(model, batch, loss, ema=None):
    model.train()
    model.optimizer.zero_grad()
    size = len(batch['rewards'])
    total = 0.0
    for start in range(0, size, micro_batch_size):
        part = {key: value[start:start + micro_batch_size] for key, value in batch.items()}
        part_loss = loss(part) / size
        part_loss.backward()
        total += part_loss.detach()
    model.optimizer.step()
    if ema is not None:
        ema.update(model)
    return float(total)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class EMA():
    # The learner process updates the weights in place while the car's process drives with them.
    # GPU work is finished before the lock is released, and the learner holds it only for the update.
    def __init__(self, model, decay):
        self.ema = ModelEmaV3(model, decay=decay)
        self.module = self.ema.module
        self.lock = context.Lock()

    def update(self, model):
        device = next(model.parameters()).device
        synchronize(device)
        with self.lock:
            self.ema.update(model)
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
