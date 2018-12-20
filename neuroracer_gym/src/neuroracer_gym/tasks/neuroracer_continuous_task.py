import time

import numpy as np

import rospy

from gym.envs.registration import register
from gym import spaces

from neuroracer_gym import neuroracer_env

register(
        id='NeuroRacer-v1',
        entry_point='neuroracer_gym:tasks.neuroracer_continuous_task.NeuroracerContinuousTask',
        # timestep_limit=timestep_limit_per_episode,
    )

class NeuroracerContinuousTask(neuroracer_env.NeuroRacerEnv):
    def __init__(self):
        self.cumulated_steps = 0.0
        self.last_action = np.zeros(2)

        self.steerin_angle_min = -1 # rospy.get_param('neuroracer_env/action_space/steerin_angle_min')
        self.steerin_angle_max = 1 # rospy.get_param('neuroracer_env/action_space/steerin_angle_max')
        self.action_space = spaces.Box(low=np.array([self.steerin_angle_min], dtype=np.float32), 
                                high=np.array([self.steerin_angle_max], dtype=np.float32))

        super(NeuroracerContinuousTask, self).__init__()

    def _set_init_pose(self):
        self.steering(0, speed=0)
        return True
    
    def _init_env_variables(self):
        self.cumulated_reward = 0.0
        self._episode_done = False
        self.last_action = np.zeros(2)

    def _compute_reward(self, observations, done):
        print(self.last_action)
        reward = 1-np.abs(self.last_action).sum()
        print(reward)

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        return reward

    def _set_action(self, action):
        steering_angle = np.clip(action, self.steerin_angle_min, self.steerin_angle_max)

        self.last_action = np.array([action, steering_angle], dtype=np.float32)
        self.steering(steering_angle, speed=10)
