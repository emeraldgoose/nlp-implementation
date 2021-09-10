""" reference : https://wikidocs.net/22660 """
import numpy as np
import pandas as pd
from torch.nn.modules.sparse import Embedding
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from make_dataset import sentence2dataset

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class CBoW(nn.Module):
    """[summary]
    Args:
        vocab ([int]): vocabulary size
        hids ([int]): hidden layer size
        window_size ([int]): sliding window size
    Param:
        V : Vocabulary size
        embedding : lookup table
        out_weights : random initialization based Gausian N(0,1)
    """
    def __init__(self, vocab, hids, window_size):
        super().__init__()
        self.hids = hids
        self.window_size = window_size
        self.V = vocab
        self.embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=hids)
        self.out_weights = nn.Linear(hids, vocab, bias=False)

    
    def forward(self, input):
        out = sum(self.embedding(input)).view(1,-1).cuda()
        out = out / (2 * self.window_size)
        out = self.out_weights(out)
        return F.softmax(out, dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    sentence = "I study math"
    words = sentence.strip().split(' ')
    vocab_size = len(words)
    window_size = 1
    hidden_layer_size = 2


    word2idx = sentence2dataset.make_dict(words) # (vocab, index)
    training_data = sentence2dataset.make_training_data(words, window_size)
    
    
    model = CBoW(vocab=vocab_size, hids=hidden_layer_size, window_size=window_size).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    model.train()
    
    epochs = 500
    for epoch in tqdm(range(epochs)):
        for i, batch in enumerate(training_data):
            context, target = batch
            target = word2idx[target]
            
            context = torch.tensor([word2idx[word] for word in context]).to(device)
            target = torch.tensor([target], dtype=torch.long).to(device)

            optimizer.zero_grad()
            output = model(context)
            
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


    for param in model.parameters():
        print(param)

if __name__ == "__main__":
    main()