#!/usr/bin/env python3

import argparse
import random
import sys

import numpy as np
import torch
from rclpy.utilities import remove_ros_args

from neuroracer_discrete import NeuroRacer
from utils import loginfo


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a NeuroRacer agent; Ctrl-C stops and saves it.')
    parser.add_argument('agent', help='dqn or ddpg')
    parser.add_argument('--output', help='Checkpoint and log directory (default: runs/<agent>); an existing checkpoint is resumed')
    parser.add_argument('--steps', type=int, help='Steps to collect in this run (default: 200000)')
    parser.add_argument('--frames', type=int, default=1, help='Camera images per state (default: 1)')
    parser.add_argument('--max-episode-steps', type=int, default=1200, help='Episode time limit (default: 1200, 2 simulated minutes)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(remove_ros_args(sys.argv if argv is None else argv)[1:])

    agent_name = args.agent
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    module = __import__(agent_name)
    agent_class = getattr(module, 'Agent')

    game = NeuroRacer(agent_class, \
                      sample_batch_size=1000, \
                      n_frames=args.frames, \
                      buffer_max_size=1000000, \
                      chunk_size=5000, \
                      add_flipped=False, \
                      env_id=getattr(module, 'env_id', 'NeuroRacer-v0'), \
                      working_dir=args.output or 'runs/' + agent_name, \
                      max_episode_steps=args.max_episode_steps)
    loginfo("Gym environment done")
    loginfo("Agent is " + agent_name)

    game.run(args.steps)


if __name__ == '__main__':
    main()
