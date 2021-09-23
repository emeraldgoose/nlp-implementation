"""
    Reference:
        https://paul-hyun.github.io/transformer-02/
"""

import torch
from torch import nn
from torch.nn.modules.normalization import LayerNorm
from Transformer.EncoderLayer import EncoderLayer
from PositionalEncoding import PositionalEncoding


class Encoder(nn.Module):
    "encoder is stack of N layers"

    def __inti__(self, config):
        super().__init__()
        self.config = config
        self.d_model = self.config.d_key * self.config.head
        self.emb = nn.Embedding(num_embeddings=self.config.vocab, embedding_dim=self.config.hidn)
        self.layers = nn.ModuleList([EncoderLayer for _ in range(self.config.N_layer)])
        self.norm = LayerNorm(EncoderLayer.size)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)
