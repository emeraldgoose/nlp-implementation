import torch

class sentence2dataset(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def one_hot_encoding(word, word2idx):
        vector = [0] * len(word2idx)
        index = word2idx[word]
        vector[index] = 1
        return vector

    def make_dict(words):
        word2idx = {} # (vocab, index)
        for vocab in words:
            if vocab not in word2idx.keys():
                word2idx[vocab] = len(word2idx)
        return word2idx

    def make_training_data(words, r): # [context_word_list, center_word]
        if r > len(words)//2: # err : out of range
            assert('r is over length of words')
        training_data = []
        for i in range(len(words)):
            target = words[i]
            context = []
            for k in range(1,r+1):
                next = i + k; prev = i - k
                if next < len(words):
                    context.append(words[next])
                if prev >= 0:
                    context.append(words[prev])
            training_data.append([context, target])
        return training_data
