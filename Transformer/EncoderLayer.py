import torch
from torch import nn
from Transformer.PosWisedFeedForward import PosWisedFeedForward
from Transformer.MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, config):
        self.config = config
        self.d_model = self.config.head * self.config.d_key
        self.attention = MultiHeadAttention(config.head, self.d_model)
        self.layerNorm1 = nn.LayerNorm(self.config.hidn)
        self.FFN = PosWisedFeedForward(config)
        self.layerNorm2 = nn.LayerNorm(self.config.hidn)
        

    def forward(self):
        pass