import numpy as np
from neuroracer_gym.neuroracer_env import NeuroRacerEnv, START_POINTS
from neuroracer_gym.tasks.neuroracer_discrete_task import NeuroRacerDiscreteTask


class Scan:
    def __init__(self, ranges):
        self.ranges = ranges


def task(ranges):
    env = NeuroRacerDiscreteTask.__new__(NeuroRacerDiscreteTask)  # No ROS connection.
    env.min_distance, env.laser_scan = .255, Scan(ranges)
    env.cumulated_reward = env.cumulated_steps = 0.0
    return env


def test_reward_clips_missing_returns_to_ten_meters():
    env = task(np.full(1081, np.inf))
    assert env._compute_reward(None, done=False) == 7.0
    assert env._compute_reward(None, done=True) == -100


def test_collision_requires_neighborhood_not_one_noisy_ray():
    scan = np.full(1081, 5.0)
    scan[540] = 0.1
    assert not task(scan)._is_collided()
    scan[530:551] = 0.1 + 0.01 * np.abs(np.arange(530, 551) - 540)  # A wall, closest at ray 540.
    assert task(scan)._is_collided()


def test_steering_command_uses_servo_speed_mapping():
    command = task(np.full(1081, 5.0))._create_steering_command(1.0, 1)
    assert np.isclose(command.linear.x, 0.5)
    assert np.isclose(command.angular.z, 0.5 * np.tan(1.0) / 0.325)


def test_episodes_start_at_the_start_points_with_random_headings():
    env = NeuroRacerEnv.__new__(NeuroRacerEnv)  # No ROS connection.
    starts = [env._random_start() for _ in range(200)]
    assert {(s['p_x'], s['p_y']) for s in starts} == set(START_POINTS)
    assert len({round(s['o_z'], 6) for s in starts}) == 200
    assert START_POINTS[env.start] == (starts[-1]['p_x'], starts[-1]['p_y'])
