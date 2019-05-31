#!/usr/bin/env python

import cv2
import numpy as np
from sklearn.utils import shuffle
from collections import deque

def preprocess(img, y_offset, x_scale, y_scale, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(cv2.cvtColor(img[y_offset:,:], cv2.COLOR_RGB2GRAY), None, fx=x_scale, fy=y_scale, interpolation=interpolation)/255.0

class Memory():
    def __init__(self, maxlen=None):
        self.action = deque(maxlen=maxlen)
        self.state = deque(maxlen=maxlen)
        self.next_state = deque(maxlen=maxlen)
        self.reward = deque(maxlen=maxlen)
        self.terminate = deque(maxlen=maxlen)
    
    def append(self, action, state, next_state, reward, terminate):
        self.action.append(action)
        self.state.append(state)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.terminate.append(terminate)
        
    def sample(self, n_samples=None):
        if not n_samples or len(self.action) <= n_samples:
            return np.array(self.action, dtype=np.int), np.array(self.state, dtype=np.float32), np.array(self.next_state, dtype=np.float32), np.array(self.reward, dtype=np.float32), np.array(self.terminate, dtype=np.bool) 
        
        action, state, next_state, reward, terminate = shuffle(self.action, self.state, self.next_state, self.reward, self.terminate, n_samples=n_samples)
        return np.array(action, dtype=np.int), np.array(state, dtype=np.float32), np.array(next_state, dtype=np.float32), np.array(reward, dtype=np.float32), np.array(terminate, dtype=np.bool) 
    
    def length(self):
        return len(self.action)
    
    def extend(self, obj):
        self.action.extend(obj.action)
        self.state.extend(obj.state)
        self.next_state.extend(obj.next_state)
        self.reward.extend(obj.reward)
        self.terminate.extend(obj.terminate)