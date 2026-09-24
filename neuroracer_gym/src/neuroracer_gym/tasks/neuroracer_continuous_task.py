import numpy as np

from gymnasium.envs.registration import register
from gymnasium import spaces

from neuroracer_gym import neuroracer_env

register(
        id='NeuroRacer-v1',
        entry_point='neuroracer_gym.tasks.neuroracer_continuous_task:NeuroracerContinuousTask',
    )

class NeuroracerContinuousTask(neuroracer_env.NeuroRacerEnv):
    def __init__(self):
        self.cumulated_steps = 0.0
        self.last_action = np.zeros(2)

        self.steerin_angle_min = -1
        self.steerin_angle_max = 1
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
        reward = 1-np.abs(self.last_action).sum()

        self.cumulated_reward += reward
        self.cumulated_steps += 1

        return float(reward)

    def _set_action(self, action):
        steering_angle = np.clip(action[0], self.steerin_angle_min, self.steerin_angle_max)

        self.last_action = np.array([action[0], steering_angle], dtype=np.float32)
        self.steering(steering_angle, speed=10)
