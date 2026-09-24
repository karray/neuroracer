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
# Largest batch per GPU pass while training: long passes would delay the driving policy's.
micro_batch_size = 32


def preprocess(img, y_offset, size, interpolation=cv2.INTER_AREA):
    """The BGR camera image without its top `y_offset` rows, as a size x size RGB image,
    channels first. uint8, so the replay buffer stores one byte per value; Normalize scales it."""
    img = cv2.resize(img[y_offset:,:], (size, size), interpolation=interpolation)
    return np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))


class H5Buffer():
    """Cyclic replay buffer in an HDF5 file, so its size is limited by disk, not RAM.

    Each preprocessed RGB frame is stored once: a row that starts an episode holds its first
    frame; every other row holds the frame after a transition, with that transition's
    action, reward and termination. `sample` reads a random contiguous block, which HDF5
    serves much faster than scattered rows, and the block rebuilds the frame stacks.
    """
    def __init__(self, state_shape, maxlen, path='buffer.hdf5', action_shape=(), action_dtype=np.ubyte):
        self.maxlen = maxlen
        self.current_idx = 0
        self.size = 0
        self.n_frames = state_shape[2]
        self.path = path
        self.lock = threading.Lock()  # The learner samples while the collector appends.

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
        """A random block of up to `n_samples` consecutive rows, and the frames before it, on `device`."""
        history = self.n_frames
        with self.lock:
            full = self.size == self.maxlen
            oldest = self.current_idx if full else 0  # Rows are counted from the oldest one.
            # Once rows are overwritten, the oldest ones lack the frames before them.
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
        # Copied to the device outside the lock, so the collector can append meanwhile.
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
    """Consecutive replay rows on the training device; states are rebuilt there from single
    frames, so batches are never copied from the host. A state stacks the RGB channels of
    its frames, oldest first: (3 * frames, height, width)."""
    def __init__(self, frames, first, actions, rewards, terminates, offset, history, device='cpu'):
        rows = np.arange(len(first))
        # Transitions are the non-first rows of the block proper; their states end one row earlier.
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
    """uint8 frames to [0, 1], as the original preprocess did, in the channels-last layout
    that makes convolutions faster on tensor cores."""
    def forward(self, states):
        return (states.float() / 255.0).contiguous(memory_format=torch.channels_last)


def autocast(device):
    """bf16 mixed precision on the GPU; weights, optimizer states and losses stay fp32."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def wait_for_gpu(tensor):
    # Keeps the learner's queue of GPU work short: CUDA queues work from all threads in
    # order, so a long queue would delay every action of the driving policy.
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()
    return tensor


def predict(model, states):
    model.eval()
    with torch.no_grad(), autocast(states.device):
        return wait_for_gpu(model(states).float())


def fit(model, block, rows, targets, batch_size, flipped=False, ema=None):
    """One shuffled epoch, like Keras' fit(states, targets, shuffle=True, epochs=1). With
    `flipped`, targets[len(rows):] belong to the mirrored states of the same rows.

    Each batch's MSE gradient is summed over micro-batches before the optimizer step: the
    same update, but in short GPU kernels between which the driving policy can run.
    `ema` follows every optimizer step."""
    model.train()
    order = torch.randperm(len(targets), device=targets.device)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        model.optimizer.zero_grad()
        loss = 0.0
        for part in index.split(micro_batch_size):
            states = block.states(rows[part % len(rows)])
            if flipped:
                mirror = part >= len(rows)
                states[mirror] = states[mirror].flip(-1)
            with autocast(states.device):
                values = model(states)
            part_loss = nn.functional.mse_loss(values.float(), targets[part], reduction='sum') / targets[index].numel()
            part_loss.backward()
            loss += wait_for_gpu(part_loss.detach())
        model.optimizer.step()
        if ema is not None:
            ema.update(model)
    return float(loss)


class EMA():
    """Exponential moving average of a model's weights (timm's ModelEmaV3), updated after
    every optimizer step. It is the target network and the network the car drives with."""
    def __init__(self, model, decay):
        self.ema = ModelEmaV3(model, decay=decay)
        self.module = self.ema.module
        self.lock = threading.Lock()  # The collector acts while the learner updates.

    def update(self, model):
        with self.lock:
            self.ema.update(model)

    def __call__(self, *inputs):
        with torch.inference_mode(), self.lock, autocast(inputs[0].device):
            return self.module(*inputs).float()


def save_checkpoint(path, **state):
    # Write then rename, so Ctrl-C during a save never leaves a truncated checkpoint.
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)
