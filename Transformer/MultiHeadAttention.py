"""
    Reference:
        http://nlp.seas.harvard.edu/2018/04/03/attention.html
        https://paul-hyun.github.io/transformer-01/
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """[summary]

    Args:
        h ([int]): number of head
        d_model ([int]): h * d_k
    """

    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dk = d_model // h
        self.n_head = h
        # query, key, value, linear layer
        self.linear = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(4)])

    def forward(self, query, key, value):
        nbatch = query.size(0)

        query, key, value = [  # (nbatch, h, d_model)
            l(x) for l, x in zip(self.linear, (query, key, value))
        ]

        # head로 나눔 (nbatch, h, h, d_k)
        query = query.view(nbatch, -1, self.n_head, self.dk)
        key = key.view(nbatch, -1, self.n_head, self.dk)
        value = value.view(nbatch, -1, self.n_head, self.dk)

        # Scaled Dot-Product
        scores = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(self.dk)
        attn = nn.Softmax(dim=-1)(scores)
        output = attn @ value

        output = output.transpose(1, 2).contiguous().view(
            nbatch, -1, self.n_head * self.dk)
        return self.linear[-1](output), attn
