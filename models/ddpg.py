# models/ddpg.py
"""
DDPG networks for CarRacing-v3.

- DDPGActor:  CNN encoder -> MLP -> 3D action in [-1, 1]
- DDPGCritic: CNN encoder + action -> MLP -> scalar Q(s, a)
"""

import torch
import torch.nn as nn
from models.common_networks import CNNFeatureExtractor


class DDPGActor(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84, action_dim=3):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels=in_channels, img_h=img_h, img_w=img_w)
        feat_dim = self.features.output_dim

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            actions in [-1, 1]^3, shape (B, 3)
        """
        feat = self.features(x)
        a = self.fc(feat)
        return self.tanh(a)


class DDPGCritic(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84, action_dim=3):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels=in_channels, img_h=img_h, img_w=img_w)
        feat_dim = self.features.output_dim

        # Q-network over [features, action]
        self.q_net = nn.Sequential(
            nn.Linear(feat_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, action):
        """
        Args:
            x:      (B, C, H, W)
            action: (B, 3) in [-1, 1]

        Returns:
            Q-value (B, 1)
        """
        feat = self.features(x)
        x_cat = torch.cat([feat, action], dim=1)
        q = self.q_net(x_cat)
        return q
