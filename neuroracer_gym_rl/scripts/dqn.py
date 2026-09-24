import os
import random

import numpy as np

import torch
from torch import nn
import timm
from timm.layers import GroupNorm

from utils import H5Buffer, Normalize, autocast, wait_for_gpu, fit, EMA, save_checkpoint, load_checkpoint, loginfo

class Agent():
    def __init__(self, state_size, action_size, buffer_max_size, chunk_size, add_flipped, working_dir='.'):
        file_name = 'dqn'+'_'+str(state_size[2])+'f'
        if add_flipped:
            file_name+='_flip'

        self.chunk_size = chunk_size
        self.add_flipped = add_flipped
        self.working_dir = working_dir
        self.weight_backup      = os.path.join(self.working_dir, file_name+'.pt')

        self.state_size         = state_size
        self.action_size        = action_size
        self.buffer             = H5Buffer(state_size, buffer_max_size, os.path.join(self.working_dir, 'buffer.hdf5'))
        self.batch_size         = 256
        self.learning_rate      = 0.0001
        self.gamma              = 0.9
        self.exploration_start  = 1.0
        self.exploration_min    = 0.01
        self.exploration_steps  = 50000
        self.ema_decay          = 0.995
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress           = {'steps': 0, 'episodes': 0}
        self.save_requested     = False
        self.loss               = None
        self.model              = self._build_model()
        self.target_model       = EMA(self.model, self.ema_decay)
        self._load_model()


    def _build_model(self):
        frames = self.state_size[2]

        model = nn.Sequential(
            Normalize(),
            timm.create_model('resnet18', pretrained=False, in_chans=3 * frames, num_classes=self.action_size,
                              norm_layer=GroupNorm),
        ).to(self.device)

        model.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loginfo(model)

        return model

    def _load_model(self):
        if os.path.isfile(self.weight_backup):
            checkpoint = load_checkpoint(self.weight_backup)
            self.model.load_state_dict(checkpoint['model'])
            self.target_model.module.load_state_dict(checkpoint['target_model'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer'])
            self.progress = checkpoint['progress']

    def save_model(self):
        self.save_requested = False
        save_checkpoint(self.weight_backup, model=self.model.state_dict(), target_model=self.target_model.module.state_dict(),
                        optimizer=self.model.optimizer.state_dict(), progress=dict(self.progress))
        loginfo("Model saved")

    @property
    def exploration_rate(self):
        fraction = min(self.progress['steps'] / self.exploration_steps, 1.0)
        return self.exploration_start + fraction * (self.exploration_min - self.exploration_start)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.target_model(torch.as_tensor(state, device=self.device))
        return int(act_values[0].argmax())

    def flip(self, batch, mirrored):
        flip = lambda images: torch.where(mirrored[:, None, None, None], images.flip(-1), images)
        return {**batch, 'actions': torch.where(mirrored, 2-batch['actions'], batch['actions']),
                'states': flip(batch['states']), 'next_states': flip(batch['next_states'])}

    def _loss(self, batch, mirrored):
        if self.add_flipped:
            batch = self.flip(batch, mirrored)
        actions, states, next_states, rewards, terminates = \
            batch['actions'].long(), batch['states'], batch['next_states'], batch['rewards'], batch['terminates']

        with torch.no_grad(), autocast(self.device):
            # Double DQN
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_pred = self.target_model.module(next_states).float().gather(1, next_actions).squeeze(1)
        targets = rewards + self.gamma * wait_for_gpu(next_pred) * ~terminates

        with autocast(self.device):
            values = self.model(states).float().gather(1, actions[:, None]).squeeze(1)
        return nn.functional.huber_loss(values, targets, reduction='sum')

    def replay(self):
        chunk_n = 2
        if self.buffer.length() < 2 * self.chunk_size:
            chunk_n = 1

        for i in range(chunk_n):
            block = self.buffer.sample(self.chunk_size, self.device)
            rows = block.transitions
            if len(rows) == 0:
                continue
            self.loss = fit(self.model, block, rows, self._loss, self.batch_size, flipped=self.add_flipped,
                            ema=self.target_model)

        if self.save_requested:
            self.save_model()
