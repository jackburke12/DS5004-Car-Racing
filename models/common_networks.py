# models/common_networks.py
"""
Stabilized convolutional feature extractor for SAC from pixels.

Adds:
    - LayerNorm after each conv layer (BatchNorm breaks actor-critic)
    - Identical API to original CNNFeatureExtractor
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84):
        super().__init__()

        # Convolution layers + LayerNorm for stability
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.ln1 = nn.LayerNorm([32, 20, 20])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.ln2 = nn.LayerNorm([64, 9, 9])
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.ln3 = nn.LayerNorm([64, 7, 7])
        self.relu3 = nn.ReLU()

        # Compute feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            x = self._forward_conv(dummy)
            self.output_dim = x.view(1, -1).shape[1]

    def _forward_conv(self, x):
        x = self.relu1(self.ln1(self.conv1(x)))
        x = self.relu2(self.ln2(self.conv2(x)))
        x = self.relu3(self.ln3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        return x.view(x.size(0), -1)
