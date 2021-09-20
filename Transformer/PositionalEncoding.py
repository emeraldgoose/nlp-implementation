"""
    Reference:
        https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
"""

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    "Position Encoding"

    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(d_model, d_model)  # (d_model, d_model)
        position = torch.arange(0, d_model).view(-1, 1)  # (d_model, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # pe : (d_model, d_model)
        pe = pe.unsqueeze(0)  # (1, d_model, d_model)
        self.register_buffer('pe', pe)  # not include model.state_dict()

    def forward(self, x):
        x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return x
