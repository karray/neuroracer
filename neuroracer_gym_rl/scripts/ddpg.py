#!/usr/bin/env python

import time
import os
import csv

import numpy as np

import rospy
import rospkg

import gym
from neuroracer_gym.tasks import neuroracer_continuous_task

# from gym.spaces import Box

# from keras import backend as k
# from keras.layers.core import Reshape
# from keras.optimizers import Adam
from keras.initializers import RandomUniform, VarianceScaling
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, concatenate, add
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import Callback as KerasCallback
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess



class Agent:
    def __init__(self, env):
        # rospack = rospkg.RosPack()
        # self.working_dir = rospack.get_path('neuroracer_gym_rl')
        # self.weight_backup      = os.path.join(self.working_dir, "neuroracer.h5")
        self.env = env

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.nb_actions  = self.env.action_space.shape[0]
        self.batch_size = 16
        self.max_buffer = 100000
        self.window_length = 16
        self.memory = SequentialMemory(limit=self.max_buffer, window_length=self.window_length)
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001
        self.gamma              = 0.9
        self.exploration_rate   = 0.95
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995

        random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.2)

        actor = self._create_actor()        
        critic, critic_action_input = self._create_critic(self.nb_actions)

        self.model = DDPGAgent(nb_actions=self.nb_actions, 
                                actor=actor, 
                                critic=critic,
                                critic_action_input=critic_action_input,
                                memory=self.memory,
                                nb_steps_warmup_critic=500,
                                nb_steps_warmup_actor=500,
                                random_process=random_process,
                                gamma=self.gamma,
                                target_model_update=.001,
                                # processor=self.processor,
                                batch_size=self.batch_size)
        self.model.compile(
            (Adam(lr=self.learning_rate_actor, clipnorm=1.), Adam(lr=self.learning_rate_critic, clipnorm=1.)),
            metrics=['mse'])


    def _create_actor(self):
        # input_shape = (self.window_length,) + self.observation_space.shape
        S = Input(shape=self.observation_space.shape)
        # S_reshape = Reshape(input_shape)(S)
        c1 = Conv2D(32, kernel_size=(4, 4), activation='relu', padding="valid",
                    kernel_initializer=VarianceScaling(mode='fan_in', distribution='uniform'))(S)
        c2 = Conv2D(32, kernel_size=(4, 4), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_in', distribution='uniform'))(c1)
        c3 = Conv2D(32, kernel_size=(4, 4), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_in', distribution='uniform'))(c2)
        c3_flatten = Flatten(name='flattened_observation')(c3)
        d1 = Dense(200, activation='relu', kernel_initializer='glorot_uniform')(c3_flatten)
        d2 = Dense(200, activation='relu', kernel_initializer='glorot_uniform')(d1)
        
        Steer = Dense(self.nb_actions,
                      activation='tanh',
                      name='prediction',
                      bias_initializer='zeros',
                      kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4)
                      )(d2)
        # Speed = Dense(1,
        #               activation='sigmoid',
        #               bias_initializer='zeros',
        #               kernel_initializer=RandomUniform(minval=0.0, maxval=3e-4)
        #               )(d2)
        # V = concatenate([Steer, Speed], name='merge_concatenate')
        
        model = Model(inputs=S, outputs=Steer)  # TODO use 'V' once multi output is supported by keras-rl

        print(model.summary())

        return model

    def _create_critic(self, nb_actions=None):
        # input_shape = (self.window_length,) + self.observation_space.shape
        S = Input(shape=self.observation_space.shape)
        # S_reshape = Reshape(input_shape)(S)
        c1 = Conv2D(32, kernel_size=(4, 4), activation='relu', padding="valid",
                    kernel_initializer=VarianceScaling(mode='fan_in', distribution='uniform'),
                    kernel_regularizer=l2(0.01))(S)
        c2 = Conv2D(32, kernel_size=(4, 4), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_in', distribution='uniform'),
                    kernel_regularizer=l2(0.01))(c1)
        c3 = Conv2D(32, kernel_size=(4, 4), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_in', distribution='uniform'),
                    kernel_regularizer=l2(0.01))(c2)
        observation_flattened = Flatten()(c3)
        O = Dense(200, activation='relu', kernel_initializer='glorot_uniform',
                  kernel_regularizer=l2(0.01))(observation_flattened)

        A = Input(shape=self.action_space.shape, name='action_input')

        # TODO activate once Speed is activated
        # a1 = Dense(200, activation='relu', kernel_initializer='glorot_uniform',
        #            kernel_regularizer=l2(0.01))(A)
        # h1 = add([O,a1], name='merge_sum')

        # TODO use upper h1 instead of this one with Speed activated
        h1 = concatenate([O, A], name='merge_concatenate')
        h2 = Dense(units=200, activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=l2(0.01))(h1)
        V = Dense(nb_actions,
                  activation='linear',
                  bias_initializer='zeros',
                  kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                  kernel_regularizer=l2(0.01)
                  )(h2)
        model = Model(inputs=[S, A], outputs=V)

        print (model.summary())

        return model, A

    def train(self, env=None, nb_steps=50000, nb_max_episode_steps=5000, nb_episodes_test=20, action_repetition=1,
              verbose=0):

        # TODO callback for save depending on loss value?
        self.model.fit(self.env,
                       nb_max_episode_steps=nb_max_episode_steps,
                       nb_steps=nb_steps,
                       action_repetition=action_repetition,
                       visualize=False)

        self.model.test(self.env,
                               nb_episodes=nb_episodes_test,
                               visualize=False,
                               nb_max_episode_steps=nb_max_episode_steps)



class StatsCallback(KerasCallback):
    def _set_env(self, env, path, interval=5000):
        self.model_file_path = os.path.join(path, 'model_checkpoint.h5')
        self.stats_file_path = os.path.join(path, 'stats_{}.csv'.format(time.time()))
        self.env = env
        self.episode_time = time.time()
        self.total_time = time.time()
        self.interval = interval
        self.total_steps = 0

    # def on_episode_begin(self, episode, logs={}):
    #     pass

    def on_episode_end(self, episode, logs={}):
        rospy.loginfo("Episode {}; steps{}; reward {}".format(episode, self.env.cumulated_steps, self.env.cumulated_reward))
        rospy.loginfo("Time {}, total {}".format(self.format_time(self.episode_time), 
                                                        self.format_time(self.total_time)))
        # with open(self.stats_file_path, newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=';')
        #     writer.writerow([episode, self.env.cumulated_steps, self.env.cumulated_reward, self.episode_time, self.total_time])

    # def on_step_begin(self, step, logs={}):
    #     pass

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            return
        # filepath = self.model_file_path.format(step=self.total_steps)
        # rospy.loginfo('Saving model to {}'.format(self.total_steps, filepath))
        self.model.save_weights(self.model_file_path, overwrite=True)
    # def on_action_begin(self, action, logs={}):
    #     pass

    # def on_action_end(self, action, logs={}):
    #     pass

    def format_time(self, t):
        m, s = divmod(int(time.time() - t), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

if __name__ == '__main__':
    rospy.init_node('neuroracer_ddpg', anonymous=True, log_level=rospy.INFO)
    
    env = gym.make('NeuroRacer-v1')
    agent = Agent(env)
    agent.train()