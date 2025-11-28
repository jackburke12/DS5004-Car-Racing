# models/dqn.py
"""
Standard DQN network for CarRacing-v3.

Architecture:
    - Shared CNN feature extractor (from common_networks)
    - Fully-connected head:
         Linear -> ReLU -> Linear (num_actions)

Used by:
    - DQN agent
    - Double DQN agent (same model, different training rule)
"""

import torch
import torch.nn as nn
from models.common_networks import CNNFeatureExtractor


class DQN(nn.Module):
    def __init__(self, num_actions, in_channels=4, img_h=84, img_w=84):
        """
        Args:
            num_actions: number of discrete actions (usually 5)
            in_channels: number of stacked frames (3 or 4)
            img_h, img_w: frame resolution
        """
        super().__init__()

        # Shared CNN feature extractor
        self.features = CNNFeatureExtractor(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w
        )

        # Fully-connected head (matches your original code)
        self.fc = nn.Sequential(
            nn.Linear(self.features.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        """
        Args:
            x: tensor (B, C, H, W)

        Returns:
            Q-values of shape (B, num_actions)
        """
        x = self.features(x)     # (B, feature_dim)
        qvals = self.fc(x)       # (B, num_actions)
        return qvals
