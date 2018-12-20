import time

import rospy

from neuroracer_gym import neuroracer_env

from gym import spaces
from gym.envs.registration import register

# timestep_limit_per_episode = 10000 # Can be any Value

default_sleep = 2

register(
        id='NeuroRacer-v0',
        entry_point='neuroracer_gym:tasks.neuroracer_discrete_task.NeuroRacerDiscreteTask',
        # timestep_limit=timestep_limit_per_episode,
    )

class NeuroRacerDiscreteTask(neuroracer_env.NeuroRacerEnv):
    def __init__(self):
        self.cumulated_steps = 0.0
        self.last_action = 1
        self.right_left = False
        self.action_space = spaces.Discrete(3)

        super(NeuroRacerDiscreteTask, self).__init__()

    def _set_init_pose(self):
        self.steering(0, speed=0)
        return True
    
    def _init_env_variables(self):
        self.cumulated_reward = 0.0
        self.last_action = 1
        self.right_left = False
        self._episode_done = False

    def _compute_reward(self, observations, done):
        reward = -0.01

        if not done:
            if self.last_action == 1:
                reward = 1
            elif self.right_left:
                reward = -1
        else:
            reward = -100

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        return reward


    def _set_action(self, action):
        steering_angle = 0
        if action == 0:
            steering_angle = -0.7
        if action == 2:
            steering_angle = 0.7

        self.right_left =  action != 1 & self.last_action != 1 & self.last_action != action

        self.last_action = action
        self.steering(steering_angle, speed=6)

