import os
import random

import numpy as np

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

# ROS packages required
import rospy
import rospkg

from utils import Memory

class Agent():
    def __init__(self, state_size, action_size, add_flipped=False, always_explore=False):
        rospack = rospkg.RosPack()
        self.add_flipped = add_flipped
        self.always_explore = always_explore
        self.working_dir = rospack.get_path('neuroracer_gym_rl')
        self.weight_backup      = os.path.join(self.working_dir, "double_dqn_8f.h5")

        self.state_size         = state_size
        self.action_size        = action_size
        self.max_buffer         = 30000
        self.memory             = Memory(self.max_buffer)
        self.learning_rate      = 0.001
        self.gamma              = 0.9
        self.exploration_rate   = 0.85
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.99
        self.model              = self._build_model()
        self.target_model = self._build_model()
        self.training_count = 0


    def _build_model(self):

        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), input_shape=self.state_size,padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2),padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(32,kernel_size=(3, 3), strides=(1, 1),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.1))      
        model.add(Dropout(0.25))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.1))      
        model.add(Dropout(0.1))

        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        model.summary()
        
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            if not self.always_explore:
                self.exploration_rate = self.exploration_min
            
        return model

    # def to_grayscale(self, img):
    #     return np.mean(img, axis=2).astype(np.uint8)

    # def downsample(self, img):
    #     return img[::2, ::2]
    


    def save_model(self):
        rospy.loginfo("Model saved") 
        self.model.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        
    def flip(self, actions, states, next_states, rewards, not_done):
        actions_flipped = 2-actions
        states_flipped = np.flip(states, axis=2)
        next_states_flipped = np.flip(next_states, axis=2)
        rewards_flipped = np.copy(rewards)
        
        next_pred_flipped = self.target_model.predict(next_states_flipped[not_done]).max(axis=1)
        rewards_flipped[not_done]+= self.gamma * next_pred_flipped
        targets_flipped = self.model.predict(states_flipped)
        targets_flipped[np.arange(len(actions_flipped)), actions_flipped] = rewards_flipped
        
        return states_flipped, targets_flipped

    def replay(self, new_data):
#         if self.memory.length() < sample_batch_size:
#             return

        rospy.loginfo("Replaying..."), 

        actions, states, next_states, rewards, terminates = new_data.sample()
        if self.memory.length() > 0:
            actions_old, states_old, next_states_old, rewards_old, terminates_old = self.memory.sample(new_data.length()*5)
            actions = np.concatenate((actions, actions_old))
            states = np.concatenate((states, states_old))
            next_states = np.concatenate((next_states, next_states_old))
            rewards = np.concatenate((rewards, rewards_old))
            terminates = np.concatenate((terminates, terminates_old))
        
        self.memory.extend(new_data)

        not_done = np.invert(terminates)
        rewards_new = np.copy(rewards)

        next_pred = self.target_model.predict(next_states[not_done]).max(axis=1)
        rewards_new[not_done]+= self.gamma * next_pred
        targets = self.model.predict(states)
        targets[np.arange(len(actions)), actions] = rewards_new
        
        if self.add_flipped:
            states_flipped, targets_flipped = self.flip(actions, states, next_states, rewards, not_done)
            states = np.concatenate((states,states_flipped))
            targets = np.concatenate((targets,targets_flipped))
            
        self.model.fit(states, targets, shuffle=True, batch_size=32, epochs=1, verbose=0)

        if self.training_count == 0 or self.training_count % 10 == 0:  
            print('Updating weights')
            self.target_model.set_weights(self.model.get_weights())
        self.training_count+=1 
        
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

        self.save_model()
        