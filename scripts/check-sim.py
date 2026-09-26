"""Integration check: sensors, control, movement, episode limits, and reset."""
import math
import numpy as np
import gymnasium as gym
from neuroracer_gym.neuroracer_env import START_POINTS
from neuroracer_gym.tasks import neuroracer_discrete_task, neuroracer_continuous_task  # registers the environments

# The starting straight is finite; eight 0.1 s steps stay well before the wall.
with gym.make('NeuroRacer-v0', max_episode_steps=8) as env:
    image, info = env.reset()
    assert env.observation_space.contains(image), (image.shape, image.dtype)
    start = info.get('position')
    for index in range(8):
        image, reward, terminated, truncated, info = env.step(1)
        assert env.observation_space.contains(image)
        assert math.isfinite(reward)
        assert not terminated, ('Unexpected collision at starting pose', info)
    assert truncated, 'Episode time limit did not truncate'
    assert start is not None and 'position' in info, 'No odometry received'
    distance = np.linalg.norm(info['position'] - start)
    assert distance > 0.2, ('Car did not move under ROS 2 control', distance)
    image, reset_info = env.reset()
    assert np.linalg.norm(reset_info['position'] - START_POINTS[reset_info['start']]) < 0.3, 'Reset did not move the car to its start point'
    print('PASS: ROS 2 RGB camera {}, lidar, {:.2f} m driving, truncation, reset'.format(image.shape, distance))

with gym.make('NeuroRacer-v1') as env:
    for steering in (0.5, -0.5):
        _, start_info = env.reset()
        for _ in range(3):  # The continuous task drives at 5 m/s.
            _, reward, terminated, _, info = env.step(np.array([steering], dtype=np.float32))
            assert math.isfinite(reward) and not terminated
        turn = math.atan2(math.sin(info['yaw'] - start_info['yaw']),
                          math.cos(info['yaw'] - start_info['yaw']))
        assert turn * steering > 0.02, ('Steering direction or response incorrect', steering, turn)
    print('PASS: continuous steering turns the car in both directions')
