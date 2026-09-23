from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .networks import QNetwork, Actor, Critic


def bootstrap_targets(rewards, terminated, next_values, gamma):
    # A time limit ends collection, not the underlying MDP: truncate != terminate.
    return rewards + gamma * (~terminated).float() * next_values


def q_next_values(online_values, target_values, double):
    if double:
        return target_values.gather(1, online_values.argmax(dim=1, keepdim=True)).squeeze(1)
    return target_values.max(dim=1).values


def tensors(batch, device):
    return {key: torch.as_tensor(value, device=device) for key, value in batch.items()}


def optimize(optimizer, loss, parameters):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(parameters, max_norm=10.0, error_if_nonfinite=True)
    optimizer.step()


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.online = QNetwork(config.frames, config.recurrent).to(self.device)
        self.target = deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.updates = 0

    def act(self, state, step=0, explore=True):
        if explore and self.rng.random() < self.config.epsilon(step):
            return int(self.rng.integers(3))
        with torch.inference_mode():
            values = self.online(torch.as_tensor(np.asarray(state)[None], device=self.device))
            return int(values.argmax(dim=1).item())

    def update(self, batch):
        batch = tensors(batch, self.device)
        with torch.no_grad():
            values = self.target(batch['next_states'])
            double = self.config.algorithm in ('double_dqn', 'double_drqn')
            online_values = self.online(batch['next_states']) if double else values
            future = q_next_values(online_values, values, double)
            targets = bootstrap_targets(batch['rewards'], batch['terminated'], future, self.config.gamma)
        predictions = self.online(batch['states']).gather(1, batch['actions'].long().view(-1, 1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        optimize(self.optimizer, loss, self.online.parameters())
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {'loss': float(loss.detach()), 'updates': self.updates}

    def state_dict(self):
        return {'online': self.online.state_dict(), 'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'updates': self.updates,
                'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.online.load_state_dict(state['online'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates = state['updates']
        self.rng.bit_generator.state = state['rng']


class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.actor = Actor(config.frames).to(self.device)
        self.critic = Critic(config.frames).to(self.device)
        self.actor_target = deepcopy(self.actor).requires_grad_(False)
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.updates = 0

    def act(self, state, step=0, explore=True):
        if explore and step < self.config.warmup:
            return self.rng.uniform(-1, 1, size=1).astype(np.float32)
        with torch.inference_mode():
            action = self.actor(torch.as_tensor(np.asarray(state)[None], device=self.device))[0].cpu().numpy()
        if explore:
            action = action + self.rng.normal(0, self.config.noise_std, size=1)
        return np.clip(action, -1, 1).astype(np.float32)

    def update(self, batch):
        batch = tensors(batch, self.device)
        with torch.no_grad():
            actions = self.actor_target(batch['next_states'])
            future = self.critic_target(batch['next_states'], actions)
            targets = bootstrap_targets(batch['rewards'], batch['terminated'], future, self.config.gamma)
        critic_loss = nn.functional.mse_loss(self.critic(batch['states'], batch['actions'].float().view(-1, 1)), targets)
        optimize(self.critic_optimizer, critic_loss, self.critic.parameters())
        # Stale critic gradients from this pass are cleared by the next critic update.
        actor_loss = -self.critic(batch['states'], self.actor(batch['states'])).mean()
        optimize(self.actor_optimizer, actor_loss, self.actor.parameters())
        with torch.no_grad():
            for target, source in ((self.actor_target, self.actor), (self.critic_target, self.critic)):
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.lerp_(param, self.config.tau)
        self.updates += 1
        return {'actor_loss': float(actor_loss.detach()), 'critic_loss': float(critic_loss.detach()), 'updates': self.updates}

    def state_dict(self):
        return {**{key: getattr(self, key).state_dict() for key in
                   ('actor', 'critic', 'actor_target', 'critic_target', 'actor_optimizer', 'critic_optimizer')},
                'updates': self.updates, 'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        for key in ('actor', 'critic', 'actor_target', 'critic_target', 'actor_optimizer', 'critic_optimizer'):
            getattr(self, key).load_state_dict(state[key])
        self.updates = state['updates']
        self.rng.bit_generator.state = state['rng']


def make_agent(config):
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    return DDPGAgent(config) if config.continuous else DQNAgent(config)
