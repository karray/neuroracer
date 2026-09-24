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
    parser.add_argument('agent', help='dqn, double_dqn, drqn, double_drqn or ddpg')
    parser.add_argument('--always-explore', type=lambda value: value.lower() == 'true', default=False,
                        help='true: keep the saved exploration rate when resuming instead of the minimum')
    parser.add_argument('--output', help='Checkpoint and log directory (default: runs/<agent>); an existing checkpoint is resumed')
    parser.add_argument('--steps', type=int, help='Steps to collect in this run (default: 200000)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(remove_ros_args(sys.argv if argv is None else argv)[1:])

    always_explore = args.always_explore
    agent_name = args.agent
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    module = __import__(agent_name)
    agent_class = getattr(module, 'Agent')

    # Frames are stored once (about 7 KB per transition), so the buffer is 7 GB on disk.
    game = NeuroRacer(agent_class, \
                      sample_batch_size=1000, \
                      n_frames=16, \
                      buffer_max_size=1000000, \
                      chunk_size=20000, \
                      add_flipped=False, \
                      always_explore=always_explore, \
                      env_id=getattr(module, 'env_id', 'NeuroRacer-v0'), \
                      working_dir=args.output or 'runs/' + agent_name)
    loginfo("Gym environment done. always_explore = " + str(always_explore))
    loginfo("Agent is " + agent_name)

    game.run(args.steps)


if __name__ == '__main__':
    main()
