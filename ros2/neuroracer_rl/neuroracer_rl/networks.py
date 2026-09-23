import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
            nn.Linear(64 * 4 * 8, 256), nn.ReLU(),
        )

    def forward(self, images):
        return self.layers(images.float() / 255.0)


class QNetwork(nn.Module):
    def __init__(self, frames, recurrent=False):
        super().__init__()
        self.recurrent = recurrent
        self.encoder = Encoder(1 if recurrent else frames)
        if recurrent:
            self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.head = nn.Linear(128 if recurrent else 256, 3)

    def forward(self, states):
        if self.recurrent:
            batch, frames, height, width = states.shape
            encoded = self.encoder(states.reshape(batch * frames, 1, height, width))
            sequence, _ = self.lstm(encoded.reshape(batch, frames, -1))
            features = sequence[:, -1]
        else:
            features = self.encoder(states)
        return self.head(features)


class Actor(nn.Module):
    def __init__(self, frames):
        super().__init__()
        self.encoder = Encoder(frames)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, states):
        return self.head(self.encoder(states))


class Critic(nn.Module):
    def __init__(self, frames):
        super().__init__()
        self.encoder = Encoder(frames)
        self.head = nn.Sequential(nn.Linear(257, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, states, actions):
        return self.head(torch.cat((self.encoder(states), actions), dim=1)).squeeze(1)
