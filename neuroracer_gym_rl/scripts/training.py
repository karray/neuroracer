#!/usr/bin/env python3

import argparse
from functools import partial
import importlib
import inspect
import json
import os
import random
import sys
import tomllib

import numpy as np
import torch
from rclpy.utilities import remove_ros_args

from neuroracer_discrete import NeuroRacer
from utils import loginfo


def resolve(target, params, *runtime):
    # All keyword arguments of `target`, defaults included; an unknown key raises TypeError.
    bound = inspect.signature(target).bind_partial(**params)
    bound.apply_defaults()
    return {key: value for key, value in bound.arguments.items() if key not in runtime}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train an agent as an experiment config describes; Ctrl-C stops and saves it.')
    parser.add_argument('config', help='Experiment config, e.g. experiments/dqn.toml; the run goes to runs/<config name>')
    parser.add_argument('--resume', action='store_true', help='Continue the run of this config')
    parser.add_argument('--drive', type=int, metavar='EPISODES',
                        help="Drive this config's run for EPISODES episodes without training or exploration")
    args = parser.parse_args(remove_ros_args(sys.argv if argv is None else argv)[1:])

    with open(args.config, 'rb') as file:
        config = tomllib.load(file)
    agent_params = dict(config['agent'])
    module, name = agent_params.pop('class').rsplit('.', 1)
    agent_class = getattr(importlib.import_module(module), name)
    agent_params = resolve(agent_class, agent_params, 'state_size', 'action_size', 'working_dir')
    training_params = resolve(NeuroRacer, config.get('training', {}), 'agent_class', 'working_dir')
    seed = config.get('seed', 0)

    # The resolved config, as saved in the run folder.
    settings = {'seed': seed, 'agent.class': config['agent']['class'],
                **{'agent.' + key: value for key, value in agent_params.items()},
                **{'training.' + key: value for key, value in training_params.items()}}

    working_dir = os.path.join('runs', os.path.splitext(os.path.basename(args.config))[0])
    settings_path = os.path.join(working_dir, 'config.json')
    if args.resume or args.drive is not None:
        if not os.path.exists(settings_path):
            sys.exit('There is no run in ' + working_dir)
        with open(settings_path) as file:
            saved = json.load(file)
        # Only the number of epochs may change: anything else is a new experiment.
        changed = sorted(key for key in saved.keys() | settings.keys()
                         if key != 'training.n_epochs' and saved.get(key) != settings.get(key))
        if changed:
            sys.exit('{} differs from {} in {}'.format(args.config, settings_path, ', '.join(changed)))
    else:
        if os.path.exists(working_dir):
            sys.exit(working_dir + ' exists; pass --resume to continue it')
        os.makedirs(working_dir)
    if args.drive is None:
        with open(settings_path, 'w') as file:
            json.dump(settings, file, indent=2)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = NeuroRacer(partial(agent_class, **agent_params), working_dir, **training_params)
    loginfo("Gym environment done")
    loginfo("Agent is " + config['agent']['class'])

    if args.drive is None:
        game.run()
    else:
        game.drive(args.drive)


if __name__ == '__main__':
    main()
