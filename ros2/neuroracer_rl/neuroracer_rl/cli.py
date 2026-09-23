import argparse
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from .config import Config, ALGORITHMS
from .runner import train, evaluate


def parser():
    root = argparse.ArgumentParser(description='PyTorch training and inference for NeuroRacer')
    commands = root.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('algorithm', nargs='?', choices=ALGORITHMS)
    training.add_argument('--steps', type=int, default=100000, help='Additional transitions to collect')
    training.add_argument('--output', type=Path)
    training.add_argument('--resume', type=Path, help='Continue from a checkpoint using its settings')
    training.add_argument('--checkpoint-interval', type=int, default=1000)
    training.add_argument('--log-interval', type=int, default=100)
    training.add_argument('--max-episode-steps', type=int, default=1000)
    for field in fields(Config):
        if field.name != 'algorithm':
            training.add_argument('--' + field.name.replace('_', '-'), type=field.type)
    evaluation = commands.add_parser('eval', help='Drive with a checkpoint, without exploration or learning')
    evaluation.add_argument('checkpoint', type=Path)
    evaluation.add_argument('--episodes', type=int, default=3)
    evaluation.add_argument('--max-episode-steps', type=int, default=1000)
    evaluation.add_argument('--device', default='cpu')
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    if args.command == 'eval':
        evaluate(args.checkpoint, episodes=args.episodes,
                 max_episode_steps=args.max_episode_steps, device=args.device)
        return
    overrides = {field.name: getattr(args, field.name) for field in fields(Config)
                 if getattr(args, field.name) is not None}
    if args.resume:
        if overrides:
            root.error('settings come from the checkpoint on --resume: ' + ', '.join(overrides))
        config, output = None, args.output or args.resume.parent
    else:
        config = Config(**overrides)
        output = args.output or Path('runs') / (
            config.algorithm + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    train(args.steps, output, config=config, resume=args.resume,
          checkpoint_interval=args.checkpoint_interval, log_interval=args.log_interval,
          max_episode_steps=args.max_episode_steps)


if __name__ == '__main__':
    main()
