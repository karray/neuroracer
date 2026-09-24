#!/usr/bin/env python3

from functools import partial
import os
import threading

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from timm.utils import ModelEmaV3

loginfo = partial(print, flush=True)
# Short GPU passes while training, so the driving policy is not delayed.
micro_batch_size = 16


def preprocess(img, y_offset, size, interpolation=cv2.INTER_AREA):
    img = cv2.resize(img[y_offset:,:], (size, size), interpolation=interpolation)
    return np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))


class H5Buffer():
    # A row that starts an episode holds its first frame; every other row holds the frame
    # after a transition, with that transition's action, reward and termination.
    def __init__(self, state_shape, maxlen, path='buffer.hdf5', action_shape=(), action_dtype=np.ubyte):
        self.maxlen = maxlen
        self.current_idx = 0
        self.size = 0
        self.n_frames = state_shape[2]
        self.path = path
        self.lock = threading.Lock()

        self.file = h5py.File(path, "w")

        def dataset(name, shape, dtype):
            return self.file.create_dataset(name, (maxlen,)+shape, dtype=dtype, chunks=(min(64, maxlen),)+shape)
        self.frames = dataset('frames', (3,) + state_shape[:2], np.uint8)
        self.first = dataset('first', (), np.bool_)
        self.actions = dataset('actions', action_shape, action_dtype)
        self.rewards = dataset('rewards', (), np.float32)
        self.terminates = dataset('terminates', (), np.bool_)

    def _write(self, frame, first, action=0, reward=0.0, terminate=False):
        with self.lock:
            idx = self.current_idx
            self.frames[idx], self.first[idx], self.actions[idx] = frame, first, action
            self.rewards[idx], self.terminates[idx] = reward, terminate
            self.current_idx = (idx + 1) % self.maxlen
            self.size = min(self.size + 1, self.maxlen)

    def start_episode(self, frame):
        self._write(frame, True)

    def append(self, action, next_frame, reward, terminate):
        self._write(next_frame, False, action, float(reward), bool(terminate))

    def sample(self, n_samples, device='cpu'):
        history = self.n_frames
        with self.lock:
            full = self.size == self.maxlen
            oldest = self.current_idx if full else 0
            low = history if full else 0
            n_samples = min(n_samples, self.size - low)
            start_idx = np.random.randint(low, self.size - n_samples + 1)
            begin_idx = max(start_idx - history, 0)

            def read(dataset):
                first = (oldest + begin_idx) % self.maxlen
                count = start_idx + n_samples - begin_idx
                if first + count <= self.maxlen:
                    return dataset[first:first + count]
                return np.concatenate((dataset[first:], dataset[:first + count - self.maxlen]))

            rows = [read(dataset) for dataset in (self.frames, self.first, self.actions, self.rewards, self.terminates)]
        return Block(*rows, start_idx - begin_idx, history, device)

    def length(self):
        return self.size

    def close(self):
        if self.file:
            self.file.close()
            os.remove(self.path)
        self.file = None

    def __del__(self):
        self.close()


class Block():
    def __init__(self, frames, first, actions, rewards, terminates, offset, history, device='cpu'):
        rows = np.arange(len(first))
        transitions = rows[offset:][~first[offset:] & (rows[offset:] > 0)]
        # Stacks repeat an episode's first frame rather than reach into the previous episode.
        episode_start = np.maximum.accumulate(np.where(first, rows, 0))
        tensor = partial(torch.as_tensor, device=device)
        self.transitions, self.frames, self.actions = tensor(transitions), tensor(frames), tensor(actions)
        self.rewards, self.terminates, self.episode_start = tensor(rewards), tensor(terminates), tensor(episode_start)
        self.back = tensor(np.arange(history - 1, -1, -1))

    def _stacks(self, ends):
        return self.frames[torch.maximum(ends[:, None] - self.back, self.episode_start[ends][:, None])].flatten(1, 2)

    def states(self, rows):
        return self._stacks(rows - 1)

    def batch(self, rows):
        return {'actions': self.actions[rows], 'states': self._stacks(rows - 1), 'next_states': self._stacks(rows),
                'rewards': self.rewards[rows], 'terminates': self.terminates[rows]}


class Normalize(nn.Module):
    def forward(self, states):
        return states.float() / 255.0


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def wait_for_gpu(tensor):
    # CUDA runs queued work in order, so a long queue would delay the driving policy.
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()
    return tensor


def fit(model, block, rows, loss, batch_size, flipped=False, ema=None):
    model.train()
    order = torch.randperm(len(rows) * (2 if flipped else 1), device=rows.device)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        model.optimizer.zero_grad()
        total = 0.0
        for part in index.split(micro_batch_size):
            part_loss = loss(block.batch(rows[part % len(rows)]), part >= len(rows)) / len(index)
            part_loss.backward()
            total += wait_for_gpu(part_loss.detach())
        model.optimizer.step()
        if ema is not None:
            ema.update(model)
    return float(total)


class EMA():
    def __init__(self, model, decay):
        self.ema = ModelEmaV3(model, decay=decay)
        self.module = self.ema.module
        self.lock = threading.Lock()

    def update(self, model):
        with self.lock:
            self.ema.update(model)

    def __call__(self, *inputs):
        with torch.inference_mode(), self.lock, autocast(inputs[0].device):
            return self.module(*inputs).float()


def save_checkpoint(path, **state):
    # Write then rename, so Ctrl-C never leaves a truncated checkpoint.
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)
