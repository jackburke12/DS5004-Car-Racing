# models/common_networks.py
"""
Shared convolutional feature extractor for CarRacing-v3 agents.

Both DQN and Dueling DQN variants use the same CNN torso:
    Conv2d -> ReLU -> Conv2d -> ReLU -> Conv2d -> ReLU

This module provides CNNFeatureExtractor, which returns:
    - forward(x): flattened feature vector
    - output_dim: number of features after flattening

Used inside DQN, Double DQN, and Dueling DQN network classes.
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84):
        """
        Args:
            in_channels: number of stacked frames (3 or 4)
            img_h, img_w: resolution of processed frames
        """
        super().__init__()

        # convolutional body (same as used in your scripts)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # compute flattened output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            conv_out = self.conv(dummy)
            self.output_dim = conv_out.view(1, -1).shape[1]

    def forward(self, x):
        """
        Args:
            x: input tensor shape (B, C, H, W)

        Returns:
            flattened features shape (B, output_dim)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x
