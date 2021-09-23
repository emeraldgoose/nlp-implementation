import torch
from torch import nn
import torch.nn.functional as F
import json


class Config(dict):
    """
        "configuration json을 읽어들이는 class"
        Reference:
            https://paul-hyun.github.io/transformer-02/u
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def main(config):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    
    
    pass


if __name__ == "__main__":
    config = Config({
        "vocab": 10,
        "hdin": 256, 
        "N_layer" : 6,
        "head" : 4,
        "d_key" : 64,
        "ffn" : 1024,
    })
    main(config)
