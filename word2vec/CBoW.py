""" reference : https://wikidocs.net/22660 """
import numpy as np
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from make_dataset import sentence2dataset

class CBoW(nn.Module):
    """[summary]
    Args:
        in_ftr ([int]): input size
        out_ftr ([int]): output size
        hids ([int]): hidden layer size
        window_size ([int]): sliding window size
    Param:
        V : Vocabulary size
        in_weights, out_weights : random initialization based Gausian N(0,1)
    """
    def __init__(self, in_ftr, out_ftr, hids, window_size):
        super().__init__()
        self.V = in_ftr
        self.hids = hids
        self.in_weights = torch.randn(in_ftr, hids, requires_grad=True)
        self.out_weights = torch.randn(hids, out_ftr, requires_grad=True)
        self.window_size = window_size
        
    
    def forward(self, context_list):
        """
        Args:
            context_list ([vector]): input vector list (one-hot vector)
        """
        # projection layer
        v = [0.]*self.hids
        v = torch.tensor(v, dtype=torch.float)
        for vector in context_list:
            index = -1
            for i in range(len(vector)):
                if vector[i]:
                    index = i
            loockup_table = self.in_weights[index]
            v += loockup_table
        v = v / (2 * self.window_size)
        # output layer
        y_hat = v @ self.out_weights
        y_hat = y_hat.reshape(self.V, 1)
        return F.softmax(y_hat, dim=1)
    
    def get_weights(self): # embedding = input weights
        return self.in_weights


def main():
    device = torch.device('cuda')

    sentence = "a b c d e f g"
    words = sentence.strip().split(' ')

    word2idx = sentence2dataset.make_dict(words) # (vocab, index)
    training_data = sentence2dataset.make_training_data(words, 2)

    # one hot encoding
    one_hot_encode = []
    for word in words:
        one_hot_encode.append(sentence2dataset.one_hot_encoding(word, word2idx))
    
    
    model = CBoW(in_ftr=len(words),out_ftr=len(words),hids=5,window_size=2).to(device)
    
    epochs = 1
    for epoch in range(epochs):
        for i, batch in enumerate(training_data):
            context, target = batch
            target = one_hot_encode[word2idx[target]]
            data = []
            for word in context:
                data.append(one_hot_encode[word2idx[word]])
            
            data = torch.tensor(data)
            target = torch.LongTensor(target)
            target = target.reshape(7,-1)
            
            output = model(data)
            
            
            # CrossEntropyLoss 안에 log_softmax가 적용되어 있으므로 output에 따로 softmax 적용하지 않음
            loss = F.cross_entropy(output, target)
            loss.backward()
            
    # print(model.get_weights)

if __name__ == "__main__":
    main()