from collections import deque
import cv2
import numpy as np


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
    def __init__(self, capacity, seed=0):
        self.data = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def append(self, state, action, reward, next_state, terminated):
        self.data.append((state, action, float(reward), next_state, bool(terminated)))

    def sample(self, size):
        # Index a list once; random access into a deque is linear in its length.
        transitions = list(self.data)
        batch = [transitions[index] for index in self.rng.choice(len(transitions), size=size, replace=False)]
        states, actions, rewards, next_states, terminated = zip(*batch)
        return {'states': np.asarray(states, dtype=np.uint8), 'actions': np.asarray(actions),
                'rewards': np.asarray(rewards, dtype=np.float32),
                'next_states': np.asarray(next_states, dtype=np.uint8),
                'terminated': np.asarray(terminated, dtype=np.bool_)}
