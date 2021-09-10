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
        out = sum(self.embedding(input)).view(1,-1)
        out = out / (2 * self.window_size)
        out = self.out_weights(out)
        return F.log_softmax(out, dim=1)


    def word_vector(self, word, word2idx):
        index = word2idx[word]
        param = next(iter(self.out_weights.parameters()))
        weights = param[index]
        return weights
                


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells."""
    
    example = sentence.strip().split(' ')
    words = []
    for word in example:
        if word == None:
            continue
        if word not in words:
            words.append(word)
    vocab_size = len(words)
    window_size = 2
    hidden_layer_size = 64


    word2idx = sentence2dataset.make_dict(words) # (vocab, index)
    training_data = sentence2dataset.make_training_data(words, window_size)
    
    
    model = CBoW(vocab=vocab_size, hids=hidden_layer_size, window_size=window_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    model.train()
    
    epochs = 1
    for epoch in tqdm(range(epochs)):
        for i, batch in enumerate(training_data):
            context, target = batch
            target = word2idx[target]
            
            context = torch.tensor([word2idx[word] for word in context]).to(device)
            target = torch.tensor([target], dtype=torch.long).to(device)

            print_context = [words[idx] for idx in context]
            print_target = words[target]
            print(print_context, print_target)

            optimizer.zero_grad()
            output = model(context)
            
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
    
    
    # TESTING
    context = ['People', 'create', 'to', 'direct']
    context_vector = torch.tensor([word2idx[word] for word in context]).to(device)
    model.eval()
    a = model(context_vector)

    #Print result
    print(f'Context: {context}\n')
    print(f'Prediction: {words[torch.argmax(a[0]).item()]}')



if __name__ == "__main__":
    main()