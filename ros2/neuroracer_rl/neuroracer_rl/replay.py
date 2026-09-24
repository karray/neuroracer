from collections import deque
from pathlib import Path
import threading
import cv2
import h5py
import numpy as np
import torch


class FrameHistory:
    """Immutable frame tuples share pixels across adjacent replay transitions."""
    def __init__(self, config):
        self.config = config
        self.frames = deque(maxlen=config.frames)

    def _frame(self, image):
        gray = cv2.cvtColor(image[self.config.crop_top:], cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(gray, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA)
        frame.setflags(write=False)
        return frame

    def reset(self, image):
        self.frames.extend([self._frame(image)] * self.config.frames)
        return tuple(self.frames)

    def append(self, image):
        self.frames.append(self._frame(image))
        return tuple(self.frames)


class ReplayBuffer:
    """Cyclic replay in an HDF5 file, so its capacity is limited by disk, not RAM.

    Rows are written in order and each preprocessed frame is stored once: a row
    that starts an episode holds its reset frame; every other row holds the frame
    after a transition with that transition's action, reward and termination.
    Training reads random contiguous blocks, which HDF5 serves much faster than
    scattered rows, and rebuilds the frame stacks from them.
    """
    def __init__(self, path, config, seed=0):
        self.path, self.config, self.capacity = Path(path), config, config.buffer_size
        self.file = h5py.File(self.path, 'w')
        def dataset(name, shape, dtype):
            return self.file.create_dataset(name, (self.capacity,) + shape, dtype,
                                            chunks=(min(64, self.capacity),) + shape)
        self.frames = dataset('frames', (config.height, config.width), np.uint8)
        self.first = dataset('first', (), np.bool_)
        self.actions = dataset('actions', (1,), np.float32) if config.continuous else dataset('actions', (), np.int64)
        self.rewards = dataset('rewards', (), np.float32)
        self.terminated = dataset('terminated', (), np.bool_)
        self.position = self.size = 0
        self.lock = threading.Lock()  # The learner reads blocks while the collector writes.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def _write(self, frame, first, action=0, reward=0.0, terminated=False):
        with self.lock:
            row = self.position
            self.frames[row], self.first[row], self.actions[row] = frame, first, action
            self.rewards[row], self.terminated[row] = reward, terminated
            self.position = (row + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def start_episode(self, frame):
        self._write(frame, True)

    def append(self, action, reward, next_frame, terminated):
        self._write(next_frame, False, action, float(reward), bool(terminated))

    def block(self, size, device='cpu'):
        """A random block of up to `size` consecutive rows and the frames before it, on `device`."""
        history = self.config.frames
        with self.lock:
            full = self.size == self.capacity
            oldest = self.position if full else 0  # Rows are counted from the oldest one.
            # Once rows are overwritten, the oldest ones lack the frames before them.
            low = history if full else 0
            size = min(size, self.size - low)
            start = int(self.rng.integers(low, self.size - size + 1))
            begin = max(start - history, 0)

            def read(dataset):
                first = (oldest + begin) % self.capacity
                count = start + size - begin
                if first + count <= self.capacity:
                    return dataset[first:first + count]
                return np.concatenate((dataset[first:], dataset[:first + count - self.capacity]))

            return Block(read(self.frames), read(self.first), read(self.actions), read(self.rewards),
                         read(self.terminated), start - begin, history, device)

    def close(self):
        self.file.close()
        self.path.unlink(missing_ok=True)


class Block:
    """Consecutive replay rows on the training device; `batch` rebuilds the frame stacks
    of some transitions there, so batches are never copied from the host."""
    def __init__(self, frames, first, actions, rewards, terminated, offset, history, device='cpu'):
        rows = np.arange(len(first))
        # Transitions are the non-first rows of the block proper; their states end one row earlier.
        self.transitions = rows[offset:][~first[offset:] & (rows[offset:] > 0)]
        # Stacks repeat an episode's first frame rather than reach into the previous episode.
        episode_start = np.maximum.accumulate(np.where(first, rows, 0))
        tensor = lambda value: torch.as_tensor(value, device=device)
        self.frames, self.actions, self.rewards = tensor(frames), tensor(actions), tensor(rewards)
        self.terminated, self.episode_start = tensor(terminated), tensor(episode_start)
        self.back, self.device = tensor(np.arange(history - 1, -1, -1)), device

    def _stacks(self, ends):
        return self.frames[torch.maximum(ends[:, None] - self.back, self.episode_start[ends][:, None])]

    def batch(self, rows):
        rows = torch.as_tensor(rows, device=self.device)
        return {'states': self._stacks(rows - 1), 'actions': self.actions[rows], 'rewards': self.rewards[rows],
                'next_states': self._stacks(rows), 'terminated': self.terminated[rows]}
