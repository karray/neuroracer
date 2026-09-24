import math
import os
import random

import numpy as np

import torch
from torch import nn

from utils import H5Buffer, micro_batch_size, Normalize, keras_init, predict, fit, ActingCopy, save_checkpoint, load_checkpoint, loginfo

class Agent():
    def __init__(self, state_size, action_size, buffer_max_size, chunk_size, add_flipped, always_explore=False, working_dir='.'):
        file_name = 'double_dqn'+'_'+str(state_size[2])+'f'
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
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress           = {'steps': 0, 'episodes': 0}
        self.save_requested     = False
        self.loss               = None
        self.model              = self._build_model()
        self.target_model = self._build_model()
        self.training_count = 0
        self.policy             = ActingCopy(self.model)


    def _build_model(self):
        height, width, frames = self.state_size

        model = nn.Sequential(
            Normalize(),
            nn.Conv2d(frames, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Dropout(0.25),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),

            nn.Flatten(),

            nn.Linear(64 * math.ceil(height / 4) * math.ceil(width / 4), 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(128, self.action_size),
        )
        model = keras_init(model).to(self.device)

        # Keras' Adam epsilon; the loss is MSE (utils.fit).
        model.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, eps=1e-7)
        loginfo(model)

        if os.path.isfile(self.weight_backup):
            checkpoint = load_checkpoint(self.weight_backup)
            model.load_state_dict(checkpoint['model'])
            model.optimizer.load_state_dict(checkpoint['optimizer'])
            self.progress = checkpoint['progress']
            if not self.always_explore:
                self.exploration_rate = self.exploration_min
            else:
                self.exploration_rate = checkpoint['exploration_rate']

        return model

    def save_model(self):
        self.save_requested = False
        save_checkpoint(self.weight_backup, model=self.model.state_dict(), optimizer=self.model.optimizer.state_dict(),
                        exploration_rate=self.exploration_rate, progress=dict(self.progress))
        loginfo("Model saved")

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.policy(torch.as_tensor(state, device=self.device))
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

        next_pred = predict(self.target_model, next_states[not_done]).max(dim=1).values
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

            # Targets come from the model before this chunk is fitted.
            targets = torch.cat([self._targets(block.batch(rows[start:start + micro_batch_size]))
                                 for start in range(0, len(rows), micro_batch_size)])

            if self.add_flipped:
                targets = torch.cat([targets] + [self._targets(self.flip(block.batch(rows[start:start + micro_batch_size])))
                                                 for start in range(0, len(rows), micro_batch_size)])
            self.loss = fit(self.model, block, rows, targets, self.batch_size, flipped=self.add_flipped)

        if self.training_count == 0 or self.training_count % 10 == 0:  
            loginfo('Updating weights')
            self.target_model.load_state_dict(self.model.state_dict())
        self.training_count+=1 

        self.policy.sync()
        if self.save_requested:
            self.save_model()
