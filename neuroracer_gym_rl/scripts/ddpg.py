#!/usr/bin/env python3

import os
import sys

import numpy as np

import torch
from torch import nn
import timm
from timm.layers import GroupNorm

from utils import H5Buffer, Normalize, EMA, autocast, save_checkpoint, load_checkpoint, loginfo

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


def resnet18(frames, outputs):
    # RGB frames; GroupNorm instead of BatchNorm, which is unstable at batch size 16.
    return timm.create_model('resnet18', pretrained=False, in_chans=3 * frames, num_classes=outputs, norm_layer=GroupNorm)


class Critic(nn.Module):
    def __init__(self, state_size, nb_actions):
        super(Critic, self).__init__()
        frames = state_size[2]
        self.observation = nn.Sequential(Normalize(), resnet18(frames, 200), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(200 + nb_actions, 200), nn.ReLU(), nn.Linear(200, 1))
        nn.init.uniform_(self.value[-1].weight, -3e-4, 3e-4)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, states, actions):
        return self.value(torch.cat((self.observation(states), actions), dim=1)).squeeze(1)


class Agent:
    def __init__(self, state_size, action_size, buffer_max_size, chunk_size, add_flipped, working_dir='.'):
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
        self.exploration_rate   = None  # Exploration is the OU noise.
        self.nb_steps_warmup    = 500
        self.ema_decay          = 0.999  # target_model_update=.001
        self.l2 = 0.01
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress           = {'steps': 0, 'episodes': 0}
        self.save_requested     = False
        self.loss               = None

        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.2)

        self.actor = self._create_actor().to(self.device)
        self.critic = self._create_critic().to(self.device)
        # Target networks, EMAs of the weights; the car drives with the target actor.
        self.target_actor = EMA(self.actor, self.ema_decay)
        self.target_critic = EMA(self.critic, self.ema_decay)
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor, eps=1e-7)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic, eps=1e-7)

        if os.path.isfile(self.weight_backup):
            checkpoint = load_checkpoint(self.weight_backup)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_actor.module.load_state_dict(checkpoint['target_actor'])
            self.target_critic.module.load_state_dict(checkpoint['target_critic'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.progress = checkpoint['progress']


    def _create_actor(self):
        backbone = resnet18(self.state_size[2], self.nb_actions)
        nn.init.uniform_(backbone.fc.weight, -3e-4, 3e-4)
        nn.init.zeros_(backbone.fc.bias)
        model = nn.Sequential(Normalize(), backbone, nn.Tanh())

        loginfo(model)

        return model

    def _create_critic(self):
        model = Critic(self.state_size, self.nb_actions)

        loginfo(model)

        return model

    def save_model(self):
        self.save_requested = False
        save_checkpoint(self.weight_backup, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        target_actor=self.target_actor.module.state_dict(), target_critic=self.target_critic.module.state_dict(),
                        actor_optimizer=self.actor.optimizer.state_dict(),
                        critic_optimizer=self.critic.optimizer.state_dict(), progress=dict(self.progress))
        loginfo("Model saved")

    def act(self, state):
        action = self.target_actor(torch.as_tensor(state, device=self.device))[0].cpu().numpy()
        action = action + self.random_process.sample()
        return np.clip(action, -1, 1).astype(np.float32)

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
            with torch.no_grad(), autocast(self.device):
                next_values = self.target_critic.module(batch['next_states'], self.target_actor.module(batch['next_states'])).float()
                targets = batch['rewards'] + self.gamma * (~batch['terminates']).float() * next_values
            with autocast(self.device):
                values = self.critic(states, actions)
            critic_loss = nn.functional.mse_loss(values.float(), targets)
            # kernel_regularizer=l2(0.01) on every critic layer
            critic_loss = critic_loss + self.l2 * sum(module.weight.pow(2).sum() for module in self.critic.modules()
                                                      if isinstance(module, (nn.Conv2d, nn.Linear)))
            self._optimize(self.critic, critic_loss)
            with autocast(self.device):
                actor_loss = -self.critic(states, self.actor(states)).float().mean()
            self._optimize(self.actor, actor_loss)
            self.target_actor.update(self.actor)
            self.target_critic.update(self.critic)
            self.loss = float(critic_loss.detach())

        if self.save_requested:
            self.save_model()


if __name__ == '__main__':
    from training import main
    main([sys.argv[0], 'ddpg'] + sys.argv[1:])
