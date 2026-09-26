#!/usr/bin/env python3

import os
import sys

import numpy as np

import torch
from torch import nn

from utils import ReplayBuffer, convnet, Normalize, EMA, autocast, to_device, save_checkpoint, load_checkpoint, loginfo

class OrnsteinUhlenbeckProcess:
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, size=1):
        self.theta, self.mu, self.sigma, self.dt, self.size = theta, mu, sigma, dt, size
        self.x_prev = np.zeros(size)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x


class Critic(nn.Module):
    def __init__(self, state_size, nb_actions):
        super(Critic, self).__init__()
        self.observation = nn.Sequential(Normalize(), convnet(state_size, 200), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(200 + nb_actions, 200), nn.ReLU(), nn.Linear(200, 1))
        nn.init.uniform_(self.value[-1].weight, -3e-4, 3e-4)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, states, actions):
        return self.value(torch.cat((self.observation(states), actions), dim=1)).squeeze(1)


class Agent:
    def __init__(self, state_size, action_size, buffer_max_size=1000000, working_dir='.', batch_size=16,
                 learning_rate_actor=0.0001, learning_rate_critic=0.001, gamma=0.9, ema_decay=0.999, l2=0.01):
        self.weight_backup      = os.path.join(working_dir, 'ddpg_{}f.pt'.format(state_size[2]))

        self.state_size = state_size
        self.nb_actions  = action_size
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(state_size, buffer_max_size, os.path.join(working_dir, 'buffer'),
                               action_shape=(action_size,), action_dtype=np.float32)
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma              = gamma
        self.exploration_rate   = None
        self.ema_decay          = ema_decay
        self.l2 = l2
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress           = {'steps': 0, 'episodes': 0, 'updates': 0}
        self.loss               = None
        self.q                  = None

        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.2)

        self.actor = self._create_actor().to(self.device)
        self.critic = self._create_critic().to(self.device)
        # The targets follow once per epoch, as many updates as a full buffer has batches, by the
        # per-update decay compounded over the epoch.
        self.updates_per_epoch = buffer_max_size // batch_size
        self.target_actor = EMA(self.actor, self.ema_decay ** self.updates_per_epoch)
        self.target_critic = EMA(self.critic, self.ema_decay ** self.updates_per_epoch)
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
        backbone = convnet(self.state_size, self.nb_actions)
        nn.init.uniform_(backbone[-1].weight, -3e-4, 3e-4)
        nn.init.zeros_(backbone[-1].bias)
        model = nn.Sequential(Normalize(), backbone, nn.Tanh())

        loginfo(model)

        return model

    def _create_critic(self):
        model = Critic(self.state_size, self.nb_actions)

        loginfo(model)

        return model

    def save_model(self):
        save_checkpoint(self.weight_backup, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        target_actor=self.target_actor.module.state_dict(), target_critic=self.target_critic.module.state_dict(),
                        actor_optimizer=self.actor.optimizer.state_dict(),
                        critic_optimizer=self.critic.optimizer.state_dict(), progress=dict(self.progress))
        loginfo("Model saved")

    def act(self, state, explore=True):
        action = self.target_actor(torch.as_tensor(state, device=self.device))[0].cpu().numpy()
        if explore:
            action = action + self.random_process.sample()
        return np.clip(action, -1, 1).astype(np.float32)

    def _optimize(self, model, loss):
        model.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        model.optimizer.step()

    def replay(self, batch):
        batch = to_device(batch, self.device)
        states, actions = batch['states'], batch['actions']
        self.actor.train()
        self.critic.train()
        with torch.no_grad(), autocast(self.device):
            next_values = self.target_critic.module(batch['next_states'], self.target_actor.module(batch['next_states'])).float()
            targets = batch['rewards'] + self.gamma * (~batch['terminates']).float() * next_values
        with autocast(self.device):
            values = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(values.float(), targets)
        critic_loss = critic_loss + self.l2 * sum(module.weight.pow(2).sum() for module in self.critic.modules()
                                                  if isinstance(module, (nn.Conv2d, nn.Linear)))
        self._optimize(self.critic, critic_loss)
        with autocast(self.device):
            actor_loss = -self.critic(states, self.actor(states)).float().mean()
        self._optimize(self.actor, actor_loss)
        self.progress['updates'] += 1
        if self.progress['updates'] % self.updates_per_epoch == 0:
            self.target_actor.update(self.actor)
            self.target_critic.update(self.critic)
        self.loss = float(critic_loss.detach())
        self.q = float(values.detach().float().mean())


if __name__ == '__main__':
    from training import main
    main([sys.argv[0], 'experiments/ddpg.toml'] + sys.argv[1:])
