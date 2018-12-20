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

n_frames = 4

class Agent():
    def __init__(self, state_size, action_size, always_explore=False):
        rospack = rospkg.RosPack()
        self.always_explore = always_explore
        self.working_dir = rospack.get_path('neuroracer_gym_rl')
        self.weight_backup      = os.path.join(self.working_dir, "neuroracer.h5")

        self.state_size         = state_size
        self.action_size        = action_size
        self.max_buffer         = 8000
        self.memory             = deque(maxlen=self.max_buffer)
        self.learning_rate      = 0.001
        self.gamma              = 0.9
        self.exploration_rate   = 0.85
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()


    def _build_model(self):

        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3), strides=(4, 4), input_shape=self.state_size,padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2),padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(16, (3, 3), strides=(3, 3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.25))

        model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.25))

        model.add(Flatten())

        # model.add(Dense(64))
        # model.add(LeakyReLU(alpha=0.1))      
        # model.add(Dropout(0.1))

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
    
#     def preprocess(self,img):
#         arr = np.array(img)
#         arr = self.downsample(np.true_divide(arr,[255.0],out=None))
# #         plt.imshow(self.downsample(arr))
# #         plt.show()
#         self.input_shape = arr.shape
#         return arr

    def save_model(self):
        rospy.loginfo("Model saved"), 
        self.brain.save(self.weight_backup)

    def act(self, history):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(np.array([np.concatenate(history, axis=2)]))
        return np.argmax(act_values[0])

    def remember(self, history, action, reward, next_history, done):
        states = np.array([np.concatenate(history, axis=2)])
        next_states = np.array([np.concatenate(next_history, axis=2)])

        self.memory.append((states, action, reward, next_states, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return

        rospy.loginfo("Replaying..."), 

        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            if not target:
                rospy.logerr("target is null: " + str(target))
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)

        
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

        self.save_model()

class NeuroRacer:
    def __init__(self, always_explore=False):
        self.sample_batch_size = 512
        self.episodes          = 50000
        self.env               = gym.make('NeuroRacer-v0')

        self.highest_reward    = -np.inf

        self.state_size        = self.env.observation_space.shape
        self.action_size       = self.env.action_space.n
        self.agent             = Agent((self.state_size[0], self.state_size[1], self.state_size[2]*n_frames), self.action_size,
                                        always_explore=always_explore)

    def format_time(self, t):
        m, s = divmod(int(time.time() - t), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def run(self):
        try:
            total_time = time.time()
            save_interval = 0

            for index_episode in range(self.episodes):
                episode_time = time.time()

                state = self.env.reset()
                
                # for skip in range(100):
                #     state, _, _, _ = self.env.step(1)

                # state = np.expand_dims(state, axis=0)

                done = False
                cumulated_reward = 0
                history = deque(maxlen=n_frames)
                next_history = deque(maxlen=n_frames)
                for i in range(n_frames):
                    history.append(state)
                    next_history.append(state)
                steps = 0
                while not done:
                    steps+=1
                    # if index_episode % 50 == 0:
                    #     self.env.render()
                    action = self.agent.act(history)

                    next_state, reward, done, _ = self.env.step(action)
                    next_history.append(next_state)
                    self.agent.remember(history, action, reward, next_history, done)
                    history.append(next_state)
                    cumulated_reward += reward

                    if save_interval > 256:
                        save_interval = 0
                        self.agent.replay(self.sample_batch_size)
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

