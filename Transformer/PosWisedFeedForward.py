"""
    Reference:
        https://paul-hyun.github.io/transformer-01/
"""

import torch
from torch import nn


class PosWisedFeedForward(nn.Module):
    "FeedForward Network"

    def __init__(self, config):
        self.config = config
        self.conv1 = nn.Conv1d(in_channels=self.config.hidn,
                               out_channels=self.config.ffn, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.ffn,
                               out_channels=self.config.hidn, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):  # shape ??
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out
