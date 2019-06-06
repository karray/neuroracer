#!/usr/bin/env python

import cv2
import h5py
import numpy as np
from sklearn.utils import shuffle
from collections import deque
import os

def preprocess(img, y_offset, x_scale, y_scale, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(cv2.cvtColor(img[y_offset:,:], cv2.COLOR_RGB2GRAY), None, fx=x_scale, fy=y_scale, interpolation=interpolation)/255.0


class H5Buffer():
    def __init__(self, state_shape, maxlen):
        self.maxlen = maxlen
        self.current_idx = 0
        
        self.file = h5py.File("buffer.hdf5", "w")
        
        self.actions = self.file.create_dataset('actions', (0,), maxshape=(maxlen,), dtype=np.ubyte)
        self.states = self.file.create_dataset('states', (0,)+state_shape, maxshape=(maxlen,)+state_shape, dtype=np.float32)
        self.next_states = self.file.create_dataset('next_states', (0,)+state_shape, maxshape=(maxlen,)+state_shape, dtype=np.float32)
        self.rewards = self.file.create_dataset('rewards', (0,), maxshape=(maxlen,), dtype=np.float32)
        self.terminates = self.file.create_dataset('terminates', (0,), maxshape=(maxlen,), dtype=np.bool)
        
    def append(self, actions, states, next_states, rewards, terminates):
        add_size = actions.shape[0]
        if self.actions.shape[0]<self.maxlen:
            self._resize(self.actions.shape[0], add_size)
            
        add_idx = add_size
        end_idx = self.current_idx + add_idx
        
        if end_idx >= self.maxlen:
            add_idx-= end_idx - self.maxlen
            end_idx = self.maxlen

        self.actions[self.current_idx:end_idx] = actions[:add_idx]
        self.states[self.current_idx:end_idx] = states[:add_idx]
        self.next_states[self.current_idx:end_idx] = next_states[:add_idx]
        self.rewards[self.current_idx:end_idx] = rewards[:add_idx]
        self.terminates[self.current_idx:end_idx] = terminates[:add_idx]
        
        self.current_idx = end_idx
        if self.current_idx == self.maxlen:
            self.current_idx = 0
        if add_idx != add_size:
            self.append(actions[add_idx:], states[add_idx:], next_states[add_idx:], rewards[add_idx:], terminates[add_idx:])
            
    def extend(self, obj):
        self.append(np.array(obj.action, dtype=np.ubyte), \
                    np.array(obj.state, dtype=np.float32), \
                    np.array(obj.next_state, dtype=np.float32), \
                    np.array(obj.reward, dtype=np.float32), \
                    np.array(obj.terminate, dtype=np.bool))
        
    def _resize(self, current_size, add_size):
        new_size = current_size + add_size
        if new_size > self.maxlen:
            new_size = self.maxlen
        self.actions.resize(new_size, axis=0)
        self.states.resize(new_size, axis=0)
        self.next_states.resize(new_size, axis=0)
        self.rewards.resize(new_size, axis=0)
        self.terminates.resize(new_size, axis=0)
        
    def sample(self, start_idx, end_idx):
#         length = self.length()
#         if length <= n_samples:
#             return self.actions[:], \
#                 self.states[:], \
#                 self.next_state[:], \
#                 self.rewards[:], \
#                 self.terminates[:] 

#         start_idx = np.random.randint(length-n_samples+1)
#         end_idx = start_idx+n_samples
        
        return self.actions[start_idx:end_idx], \
                self.states[start_idx:end_idx], \
                self.next_states[start_idx:end_idx], \
                self.rewards[start_idx:end_idx], \
                self.terminates[start_idx:end_idx]
                
    def length(self):
        return len(self.actions)
    
    def close(self):
        if self.file:
            self.file.close()
            os.remove('buffer.hdf5')
        self.file = None
        
    def __del__(self):
        self.close()    

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