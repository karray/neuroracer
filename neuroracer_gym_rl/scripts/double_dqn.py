#!/usr/bin/env python

from collections import deque
import time
import random
import os

import numpy as np
import gym


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

from neuroracer_gym.tasks import neuroracer_discrete_task

      
class Agent():
    def __init__(self, state_size, action_size, always_explore=False):
        rospack = rospkg.RosPack()
        self.always_explore = always_explore
        self.working_dir = rospack.get_path('neuroracer_gym_rl')
        self.weight_backup      = os.path.join(self.working_dir, "double_dqn.h5")

        self.state_size         = state_size
        self.action_size        = action_size
        self.max_buffer         = 30000
        self.memory             = Memory(self.max_buffer)
        self.learning_rate      = 0.001
        self.gamma              = 0.99
        self.exploration_rate   = 0.85
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.99
        self.model              = self._build_model()
        self.target_model = self._build_model()


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
        rospy.loginfo("Model saved"), 
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

    def replay(self, new_data, add_flipped=True):
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
        
        if add_flipped:
            states_flipped, targets_flipped = self.flip(actions, states, next_states, rewards, not_done)
            states = np.concatenate((states,states_flipped))
            targets = np.concatenate((targets,targets_flipped))
            
        self.target_model.set_weights(self.model.get_weights())
        
        self.model.fit(states, targets, shuffle=True, batch_size=256, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

        self.save_model()

class NeuroRacer:
    def __init__(self, always_explore=False):
        self.sample_batch_size = 5000
        self.episodes          = 100000
        self.env               = gym.make('NeuroRacer-v0')

        self.highest_reward    = -np.inf
        
        self.n_frames = 4

        self.img_y_offset = 200
        self.img_y_scale = 0.2
        self.img_x_scale = 0.2

        state_size = self.env.observation_space.shape
        self.state_size        = (int((state_size[0]-self.img_y_offset)*self.img_y_scale), int(state_size[1]*self.img_x_scale), 1)
        rospy.loginfo("State size")
        rospy.loginfo(self.state_size)

        self.action_size       = self.env.action_space.n
        self.agent             = Agent((self.state_size[0], self.state_size[1], self.state_size[2]*self.n_frames), self.action_size, always_explore=always_explore)

        
    def format_time(self, t):
        m, s = divmod(int(time.time() - t), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)
    

    
    def run(self):
        try:
            total_time = time.time()
            save_interval = 0
            
            memory = Memory()

            for index_episode in range(self.episodes):
                episode_time = time.time()

                self.env.initial_position = {'p_x': np.random.uniform(1,4), 'p_y': 3.7, 'p_z': 0.05, 'o_x': 0, 'o_y': 0.0, 'o_z': np.random.uniform(0.4,1), 'o_w': 0.855}
                state = self.env.reset()
                state = self.preprocess(state)
#                 state = np.expand_dims(state, axis=0)

                done = False
                cumulated_reward = 0
                
                stacked_states = deque(maxlen=self.n_frames)
                stacked_next_states = deque(maxlen=self.n_frames)
                for i in range(self.n_frames):
                    stacked_states.append(state)
                    stacked_next_states.append(state)
                
                steps = 0
                while not done:
                    steps+=1
                    # if index_episode % 50 == 0:
                    #     self.env.render()
                    
                    action = self.agent.act(np.expand_dims(np.stack(stacked_states, axis=2), axis=0))
#                     action = self.agent.act(state)


                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess(next_state)
#                     next_state = np.expand_dims(next_state, axis=0)
                    stacked_next_states.append(next_state)

                    memory.append(action,np.stack(stacked_states, axis=2), np.stack(stacked_next_states, axis=2), reward, done)        
#                     self.agent.remember(history, action, reward, next_history, done)
                    stacked_states.append(next_state)
#                     self.agent.remember(state, action, reward, next_state, done)
#                     state = next_state
                    
                    cumulated_reward += reward

                    if save_interval >= self.sample_batch_size:
                        save_interval = 0
                        replay_time = time.time()
                        self.agent.replay(memory)
                        rospy.loginfo("Replay time {}".format(time.time()-replay_time))
                        memory = Memory()
                        # rospy.loginfo("Episode {} of {}. In-episode training".format(index_episode, self.episodes))
                        # rospy.loginfo("Step {}, reward {}/{}".format(steps, cumulated_reward, self.highest_reward))
                        # rospy.loginfo("Episode time {}, total {}".format(self.format_time(episode_time), 
                        #                                                 self.format_time(total_time)))
                    save_interval+=1
                    

                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                rospy.loginfo("Episode {} of {}".format(index_episode, self.episodes))
                rospy.loginfo("total steps {}, reward {}/{}".format(steps, cumulated_reward, self.highest_reward))
                rospy.loginfo("Episode time {}, total {}".format(self.format_time(episode_time), 
                                                                self.format_time(total_time)))
                rospy.loginfo("exploration_rate {}".format(self.agent.exploration_rate))
                
                # self.agent.replay(self.sample_batch_size)
                # if index_episode % 50 == 0:
                #     self.env.close()
        finally:
            self.env.close()
            # self.agent.save_model()
            # print("Best score: " + str(self.best_score))

if __name__ == '__main__':

    rospy.init_node('neuroracer_qlearn', anonymous=True, log_level=rospy.INFO)

    always_explore = rospy.get_param('/neuroracer_gym/always_explore')
    # Create the Gym environment
    game = NeuroRacer(always_explore=always_explore)
    rospy.logwarn("Gym environment done. always_explore = " + str(always_explore))

    # Set the logging system
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('demo')
    # outdir = pkg_path + '/training_results'
    # rospy.loginfo("Monitor Wrapper started")

    game.run()

