"""
    Referenece:
        https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """[summary]

    Args:
        input_dimension ([int]): input dimension
        key_dimension ([int]): d_k
        value_dimension ([int]): d_v
    Params:
        key_dimension ([int]): d_k
        query_weight: linear transform weights
        key_weight: linear transform weights
        value_weight: linear transform weights
    """

    def __init__(self, input_dimension, key_dimension, value_dimension):
        super().__init__()
        self.key_dimension = key_dimension
        self.query_weight = nn.Linear(
            in_features=input_dimension, out_features=key_dimension, bias=False)
        self.key_weight = nn.Linear(
            in_features=input_dimension, out_features=key_dimension, bias=False)
        self.value_weight = nn.Linear(
            in_features=input_dimension, out_features=value_dimension, bias=False)

    def forward(self, inputs):
        querys = self.query_weight(inputs)
        keys = self.key_weight(inputs)
        values = self.value_weight(inputs)

        # (number_of_inputs, number_of_inputs)
        scaled = (querys @ keys.T) / np.math.sqrt(self.key_dimension)
        value_weight = F.softmax(scaled, dim=1)  # row-wised softmax
        # 여기서부터 참고함
        weighted = values[:, None] * value_weight.T[:, :, None]
        return weighted.sum(dim=0)


def main():
    number_of_input_vector = 3
    input_dimension = 3
    input_vectors = np.array([np.random.randn(input_dimension)
                             for _ in range(number_of_input_vector)])
    input_vectors = torch.tensor(input_vectors, dtype=torch.float)

    model = SelfAttention(input_dimension=input_dimension,
                          key_dimension=3, value_dimension=3)

    output = model(input_vectors)
    print(output)


if __name__ == "__main__":
    main()
