# models/ddpg.py
"""
DDPG / TD3 networks for CarRacing-v3 with stabilized CNN encoders.

Fixes:
    - Separate CNNs for actor & critic
    - LayerNorm inside CNN (from CNNFeatureExtractor)
    - Stop-gradient for actor CNN features (prevents destabilizing critic)
"""

import torch
import torch.nn as nn
from models.common_networks import CNNFeatureExtractor


class DDPGActor(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84, action_dim=3):
        super().__init__()

        # Actor has its own encoder
        self.actor_features = CNNFeatureExtractor(in_channels, img_h, img_w)
        feat_dim = self.actor_features.output_dim

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
        x: (B, C, H, W) normalized image frames
        returns: actions in [-1,1]
        """

        # Very important: actor sees **stop-gradient** features
        feat = self.actor_features(x).detach()

        a = self.fc(feat)
        return self.tanh(a)


class DDPGCritic(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84, action_dim=3):
        super().__init__()

        # Critic has its own encoder (no stop-grad)
        self.critic_features = CNNFeatureExtractor(in_channels, img_h, img_w)
        feat_dim = self.critic_features.output_dim

        self.q_net = nn.Sequential(
            nn.Linear(feat_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, action):
        """
        x: (B,C,H,W)
        action: (B,3)
        returns: Q-value
        """
        feat = self.critic_features(x)    # critic learns visual features
        x_cat = torch.cat([feat, action], dim=1)
        return self.q_net(x_cat)
    
    
    def forward_actor(self, x, action):
        """
        Used only for actor update.
        Detaches critic CNN features to prevent interference.
        """
        feat = self.critic_features(x).detach()
        x_cat = torch.cat([feat, action], dim=1)
        return self.q_net(x_cat)
