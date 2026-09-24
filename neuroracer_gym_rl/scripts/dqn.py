import os
import random

import numpy as np

import torch
from torch import nn
import timm

from utils import H5Buffer, micro_batch_size, Normalize, predict, fit, EMA, save_checkpoint, load_checkpoint, loginfo

class Agent():
    def __init__(self, state_size, action_size, buffer_max_size, chunk_size, add_flipped, always_explore=False, working_dir='.'):
        file_name = 'dqn'+'_'+str(state_size[2])+'f'
        if add_flipped:
            file_name+='_flip'

        self.chunk_size = chunk_size
        self.add_flipped = add_flipped
        self.always_explore = always_explore
        self.working_dir = working_dir
        self.weight_backup      = os.path.join(self.working_dir, file_name+'.pt')

        self.state_size         = state_size
        self.action_size        = action_size
        self.buffer             = H5Buffer(state_size, buffer_max_size, os.path.join(self.working_dir, 'buffer.hdf5'))
        self.batch_size         = 1000
        self.learning_rate      = 0.001
        self.gamma              = 0.9
        self.exploration_rate   = 0.85
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.99
        self.ema_decay          = 0.995
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress           = {'steps': 0, 'episodes': 0}
        self.save_requested     = False
        self.loss               = None
        self.model              = self._build_model()
        # The target network, an EMA of the model's weights; the car drives with it.
        self.target_model       = EMA(self.model, self.ema_decay)
        self._load_model()


    def _build_model(self):
        frames = self.state_size[2]

        # Standard resnet18: average pooling and a linear head to the Q-values.
        model = nn.Sequential(
            Normalize(),
            timm.create_model('resnet18', pretrained=False, in_chans=3 * frames, num_classes=self.action_size),
        ).to(self.device, memory_format=torch.channels_last)

        # Keras' Adam epsilon; the loss is MSE (utils.fit).
        model.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, eps=1e-7)
        loginfo(model)

        return model

    def _load_model(self):
        if os.path.isfile(self.weight_backup):
            checkpoint = load_checkpoint(self.weight_backup)
            self.model.load_state_dict(checkpoint['model'])
            self.target_model.module.load_state_dict(checkpoint['target_model'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer'])
            self.progress = checkpoint['progress']
            if not self.always_explore:
                self.exploration_rate = self.exploration_min
            else:
                self.exploration_rate = checkpoint['exploration_rate']

    def save_model(self):
        self.save_requested = False
        save_checkpoint(self.weight_backup, model=self.model.state_dict(), target_model=self.target_model.module.state_dict(),
                        optimizer=self.model.optimizer.state_dict(),
                        exploration_rate=self.exploration_rate, progress=dict(self.progress))
        loginfo("Model saved")

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.target_model(torch.as_tensor(state, device=self.device))
        return int(act_values[0].argmax())

    def update_exploration(self):
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def flip(self, batch):
        return {**batch, 'actions': 2-batch['actions'],
                'states': batch['states'].flip(-1), 'next_states': batch['next_states'].flip(-1)}

    def _targets(self, batch):
        actions, states, next_states, rewards, terminates = \
            batch['actions'].long(), batch['states'], batch['next_states'], batch['rewards'], batch['terminates']

        not_done = ~terminates
        rewards_new = rewards.clone()

        next_pred = predict(self.target_model.module, next_states[not_done]).max(dim=1).values
        rewards_new[not_done]+= self.gamma * next_pred
        targets = predict(self.model, states)
        targets[torch.arange(len(actions), device=actions.device), actions] = rewards_new
        return targets

    def replay(self):
        chunk_n = 2
        if self.buffer.length() < 2 * self.chunk_size:
            chunk_n = 1

        for i in range(chunk_n):
            block = self.buffer.sample(self.chunk_size, self.device)
            rows = block.transitions
            if len(rows) == 0:
                continue

            # Targets are computed once, before this chunk is fitted.
            targets = torch.cat([self._targets(block.batch(rows[start:start + micro_batch_size]))
                                 for start in range(0, len(rows), micro_batch_size)])

            if self.add_flipped:
                targets = torch.cat([targets] + [self._targets(self.flip(block.batch(rows[start:start + micro_batch_size])))
                                                 for start in range(0, len(rows), micro_batch_size)])
            self.loss = fit(self.model, block, rows, targets, self.batch_size, flipped=self.add_flipped,
                            ema=self.target_model)

        if self.save_requested:
            self.save_model()
