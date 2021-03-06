import os
import random
import time

import numpy as np

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
# from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

# ROS packages required
import rospy
import rospkg

from utils import Memory, H5Buffer

class Agent():
    def __init__(self, state_size, action_size, buffer_max_size, chunk_size, add_flipped, always_explore=False):
        file_name = 'double_drqn'+'_'+str(state_size[2])+'f'
        if add_flipped:
            file_name+='_flip'
        rospack = rospkg.RosPack()
        
        self.model_batch = 128
        
        self.chunk_size = chunk_size
        self.add_flipped = add_flipped
        self.always_explore = always_explore
        self.working_dir = rospack.get_path('neuroracer_gym_rl')
        self.weight_backup      = os.path.join(self.working_dir, file_name+'.h5')

        self.state_size         = (state_size[2], state_size[0], state_size[1], 1)
        self.action_size        = action_size
        self.buffer             = H5Buffer(state_size, buffer_max_size)
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
        model.add(TimeDistributed(Conv2D(16, kernel_size=3, strides=1, padding='same'), input_shape=self.state_size))
        model.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        model.add(TimeDistributed(MaxPooling2D((2, 2),padding='same')))
        model.add(TimeDistributed(Dropout(0.25)))

        model.add(TimeDistributed(Conv2D (32, kernel_size=3, strides=1, padding='same')))
        model.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        model.add(TimeDistributed(Dropout(0.25)))

        model.add(TimeDistributed(Conv2D (64, kernel_size=3, strides=2, padding='same')))
        model.add(TimeDistributed(LeakyReLU(alpha=0.1)))
        model.add(TimeDistributed(Dropout(0.25)))

        model.add(TimeDistributed(Flatten()))

        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        model.add(LSTM(512,  activation='tanh'))
        model.add(Dense(output_dim=self.action_size, activation='linear'))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            if not self.always_explore:
                self.exploration_rate = self.exploration_min
            
        return model


    def save_model(self):
        rospy.loginfo("Model saved") 
        self.model.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        
        state = np.expand_dims(np.moveaxis(state, -1, 1), axis=-1)

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
        rospy.loginfo("Replaying..."), 

        self.buffer.extend(new_data)
        buffer_length = self.buffer.length()
        
        chunks = buffer_length / self.chunk_size
        
        chunk_n = 2
        if chunks < 2:
            chunk_n = 1
            chunks=1
            
        print('buffer length', buffer_length)
        print('chunks', chunks)
        
        for i in np.random.choice(range(chunks), chunk_n, False):
            print('fitting', i)
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size
            
            loading_time = time.time()
            actions, states, next_states, rewards, terminates = self.buffer.sample(start_idx, end_idx)
            print('loading {} samples time: {}'.format(self.chunk_size, time.time()-loading_time))

            states = np.expand_dims(np.moveaxis(states, -1, 1), axis=-1)
            next_states = np.expand_dims(np.moveaxis(next_states, -1, 1), axis=-1)
            print('================== shapes:')
            print(states.shape,next_states.shape)
            
            not_done = np.invert(terminates)
            rewards_new = np.copy(rewards)

            tmp_pred = self.target_model.predict(next_states[not_done], batch_size=self.model_batch)
                
            next_pred = tmp_pred.max(axis=1)
            rewards_new[not_done]+= self.gamma * next_pred
            targets = self.model.predict(states)
            targets[np.arange(len(actions)), actions] = rewards_new

            if self.add_flipped:
                states_flipped, targets_flipped = self.flip(actions, states, next_states, rewards, not_done)
                states = np.concatenate((states,states_flipped))
                targets = np.concatenate((targets,targets_flipped))
                        
            fit_time = time.time()
            self.model.fit(states, targets, shuffle=True, batch_size=self.model_batch, epochs=1, verbose=0)
            print('fit time:', time.time()-fit_time)
            
        if self.training_count == 0 or self.training_count % 10 == 0:  
            print('Updating weights')
            self.target_model.set_weights(self.model.get_weights())
        self.training_count+=1 
        
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

        self.save_model()
        