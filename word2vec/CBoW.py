""" 
    Reference : 
        https://wikidocs.net/22660 
    
    Test Reference:
        https://github.com/FraLotito/pytorch-continuous-bag-of-words
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from make_dataset import sentence2dataset

class CBoW(nn.Module):
    """[summary]
    Args:
        vocab ([int]): vocabulary size
        embedding_size ([int]): embedding layer size
        window_size ([int]): sliding window size
    Param:
        embedding : lookup table
        out_weights : 
    functions:
        __init__() : initialization
        forward() : feedforward
        word_vector() : return word embedding vector
    """
    def __init__(self, vocab, embedding_size, window_size):
        super().__init__()
        self.hids = embedding_size
        self.window_size = window_size
        self.embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=embedding_size)
        self.out_weights = nn.Linear(embedding_size, vocab, bias=False)
    
    def forward(self, input):
        out = sum(self.embedding(input)).view(1,-1)
        out = out / (2 * self.window_size)
        out = self.out_weights(out)
        return F.log_softmax(out, dim=1)

    def word_vector(self, word, word2idx):
        word = torch.tensor(word2idx[word])
        return self.embedding(word).view(1,-1)


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
    embedding_size = 64

    word2idx, idx2word = sentence2dataset.make_dict(words) # (vocab, index)
    training_data = sentence2dataset.make_training_data(words, window_size)
    
    model = CBoW(vocab=vocab_size, embedding_size=embedding_size, window_size=window_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    model.train()
    
    epochs = 1
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

    # TESTING
    context = ['People', 'create', 'to', 'direct']
    context_vector = torch.tensor([word2idx[word] for word in context]).to(device)
    model.eval()
    a = model(context_vector)

    #Print result
    print(f'Context: {context}')
    print(f'Prediction: {idx2word[torch.argmax(a[0]).item()]}')


if __name__ == "__main__":
    main()