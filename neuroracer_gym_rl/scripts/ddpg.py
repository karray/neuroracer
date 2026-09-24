#!/usr/bin/env python3

import math
import os
import sys

import numpy as np

import torch
from torch import nn

from utils import H5Buffer, Normalize, keras_init, ActingCopy, save_checkpoint, load_checkpoint, loginfo

env_id = 'NeuroRacer-v1'


class OrnsteinUhlenbeckProcess:
    """keras-rl's OrnsteinUhlenbeckProcess."""
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, size=1):
        self.theta, self.mu, self.sigma, self.dt, self.size = theta, mu, sigma, dt, size
        self.x_prev = np.zeros(size)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x


def conv_layers(frames):
    layers = []
    for channels in (frames, 32, 32):
        conv = nn.Conv2d(channels, 32, kernel_size=4)
        # VarianceScaling(mode='fan_in', distribution='uniform')
        nn.init.uniform_(conv.weight, -math.sqrt(3 / (channels * 16)), math.sqrt(3 / (channels * 16)))
        nn.init.zeros_(conv.bias)
        layers += [conv, nn.ReLU()]
    return layers


class Critic(nn.Module):
    def __init__(self, state_size, nb_actions):
        super(Critic, self).__init__()
        height, width, frames = state_size
        dense = keras_init(nn.Sequential(nn.Flatten(), nn.Linear(32 * (height - 9) * (width - 9), 200), nn.ReLU()))
        self.observation = nn.Sequential(Normalize(), *conv_layers(frames), *dense)
        self.value = keras_init(nn.Sequential(nn.Linear(200 + nb_actions, 200), nn.ReLU(), nn.Linear(200, 1)))
        nn.init.uniform_(self.value[-1].weight, -3e-4, 3e-4)

    def forward(self, states, actions):
        return self.value(torch.cat((self.observation(states), actions), dim=1)).squeeze(1)


class Agent:
    def __init__(self, state_size, action_size, buffer_max_size, chunk_size, add_flipped, always_explore=False, working_dir='.'):
        self.weight_backup      = os.path.join(working_dir, 'ddpg_{}f.pt'.format(state_size[2]))

        self.state_size = state_size
        self.nb_actions  = action_size
        self.chunk_size = chunk_size
        self.batch_size = 16
        self.buffer = H5Buffer(state_size, buffer_max_size, os.path.join(working_dir, 'buffer.hdf5'),
                               action_shape=(action_size,), action_dtype=np.float32)
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001
        self.gamma              = 0.9
        self.exploration_rate   = 0.95  # Unused, as in the original: exploration is the OU noise.
        self.nb_steps_warmup    = 500
        self.target_model_update = .001
        self.l2 = 0.01
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress           = {'steps': 0, 'episodes': 0}
        self.save_requested     = False
        self.loss               = None

        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.2)

        self.actor = self._create_actor().to(self.device)
        self.critic = self._create_critic().to(self.device)
        self.target_actor = self._create_actor().to(self.device)
        self.target_critic = self._create_critic().to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor, eps=1e-7)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic, eps=1e-7)

        if os.path.isfile(self.weight_backup):
            checkpoint = load_checkpoint(self.weight_backup)
            for name in ('actor', 'critic', 'target_actor', 'target_critic'):
                getattr(self, name).load_state_dict(checkpoint[name])
            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.progress = checkpoint['progress']
        self.policy = ActingCopy(self.actor)


    def _create_actor(self):
        height, width, frames = self.state_size
        dense = keras_init(nn.Sequential(nn.Flatten(), nn.Linear(32 * (height - 9) * (width - 9), 200), nn.ReLU(),
                                         nn.Linear(200, 200), nn.ReLU()))
        steer = nn.Linear(200, self.nb_actions)
        nn.init.uniform_(steer.weight, -3e-4, 3e-4)
        nn.init.zeros_(steer.bias)
        model = nn.Sequential(Normalize(), *conv_layers(frames), *dense, steer, nn.Tanh())

        loginfo(model)

        return model

    def _create_critic(self):
        model = Critic(self.state_size, self.nb_actions)

        loginfo(model)

        return model

    def save_model(self):
        self.save_requested = False
        save_checkpoint(self.weight_backup, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        target_actor=self.target_actor.state_dict(), target_critic=self.target_critic.state_dict(),
                        actor_optimizer=self.actor.optimizer.state_dict(),
                        critic_optimizer=self.critic.optimizer.state_dict(), progress=dict(self.progress))
        loginfo("Model saved")

    def act(self, state):
        action = self.policy(torch.as_tensor(state, device=self.device))[0].cpu().numpy()
        action = action + self.random_process.sample()
        return np.clip(action, -1, 1).astype(np.float32)

    def update_exploration(self):
        pass  # The OU noise does not decay.

    def _optimize(self, model, loss):
        model.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)  # clipnorm=1.
        model.optimizer.step()

    def replay(self):
        if self.buffer.length() < self.nb_steps_warmup:
            return False
        block = self.buffer.sample(self.chunk_size, self.device)
        rows = block.transitions[torch.randperm(len(block.transitions), device=self.device)]
        self.actor.train()
        self.critic.train()
        for start in range(0, len(rows), self.batch_size):
            batch = block.batch(rows[start:start + self.batch_size])
            states, actions = batch['states'], batch['actions']
            with torch.no_grad():
                next_values = self.target_critic(batch['next_states'], self.target_actor(batch['next_states']))
                targets = batch['rewards'] + self.gamma * (~batch['terminates']).float() * next_values
            critic_loss = nn.functional.mse_loss(self.critic(states, actions), targets)
            # kernel_regularizer=l2(0.01) on every critic layer
            critic_loss = critic_loss + self.l2 * sum(parameter.pow(2).sum() for name, parameter
                                                      in self.critic.named_parameters() if name.endswith('weight'))
            self._optimize(self.critic, critic_loss)
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self._optimize(self.actor, actor_loss)
            with torch.no_grad():
                for target, source in ((self.target_actor, self.actor), (self.target_critic, self.critic)):
                    for target_param, param in zip(target.parameters(), source.parameters()):
                        target_param.lerp_(param, self.target_model_update)
            self.loss = float(critic_loss.detach())

        self.policy.sync()
        if self.save_requested:
            self.save_model()


if __name__ == '__main__':
    from training import main
    main([sys.argv[0], 'ddpg'] + sys.argv[1:])
