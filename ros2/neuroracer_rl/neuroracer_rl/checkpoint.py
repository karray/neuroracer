from dataclasses import asdict, replace
from pathlib import Path
import os
import torch
from .config import Config
from .agents import make_agent


def save_checkpoint(path, agent, step, episodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'config': asdict(agent.config), 'agent': agent.state_dict(), 'step': int(step),
               'episodes': int(episodes), 'torch_rng': torch.get_rng_state()}
    # Write then rename, so Ctrl-C during a save never leaves a truncated checkpoint.
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device=None):
    # State dictionaries and primitive metadata only; no pickled model objects.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    config = Config(**payload['config'])
    agent = make_agent(replace(config, device=device) if device else config)
    agent.load_state_dict(payload['agent'])
    torch.set_rng_state(payload['torch_rng'])
    return agent, payload['step'], payload['episodes']
