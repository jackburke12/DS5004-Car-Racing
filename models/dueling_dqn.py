# models/dueling_dqn.py
"""
Dueling DQN network for CarRacing-v3.

Architecture:
    - Shared CNN feature extractor (common_networks.py)
    - Two fully-connected streams:
        * Value stream      -> V(s)
        * Advantage stream  -> A(s, a)
    - Combine them as:
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,*)))

"""

import torch
import torch.nn as nn
from models.common_networks import CNNFeatureExtractor


class DuelingDQN(nn.Module):
    def __init__(self, num_actions, in_channels=3, img_h=84, img_w=84):
        """
        Args:
            num_actions: number of discrete actions (usually 5)
            in_channels: number of stacked frames
            img_h, img_w: preprocessed frame size
        """
        super().__init__()

        # Shared CNN feature extractor (same torso as DQN)
        self.features = CNNFeatureExtractor(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w
        )

        feat_dim = self.features.output_dim

        # Value stream
        self.val_fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream
        self.adv_fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
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
        features = self.features(x)  # (B, feat_dim)

        values = self.val_fc(features)          # (B, 1)
        advantages = self.adv_fc(features)      # (B, num_actions)

        # Q = V + (A - mean(A))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals
