"""
    Reference :
        https://wikidocs.net/22660
        https://reniew.github.io/22/
        https://www.kaggle.com/karthur10/skip-gram-implementation-with-pytorch-step-by-step
        https://velog.io/@jaeyun95/NLP%EC%8B%A4%EC%8A%B51.%EC%9E%90%EC%97%B0%EC%96%B4-%EC%B2%98%EB%A6%AC-%EA%B0%9C%EC%9A%94-%EB%8B%A8%EC%96%B4-%EC%9E%84%EB%B2%A0%EB%94%A9
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from make_dataset import sentence2dataset

class SkipGram(nn.Module):
    def __init__(self, vocab, embedding_size, window_size, device):
        super().__init__()
        self.window_size = window_size # context_size = 2 * window_size
        self.vocab = vocab
        self.in_weight = nn.Linear(in_features=vocab, out_features=embedding_size, bias=False)
        self.out_weight = nn.Linear(in_features=embedding_size, out_features=vocab*2*window_size, bias=False)

    def forward(self, center):
        out = self.in_weight(center)
        out = self.out_weight(out).view(1,-1)
        out = F.log_softmax(out, dim=1)
        return out.view(2*self.window_size, self.vocab)

    # def word_vector(self, word, word2idx):
    #     one_hot = sentence2dataset.one_hot_encoding(word, word2idx).view(-1,1)
    #     return self.out_weight(one_hot)
        


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
    embedding_size = 5

    word2idx, idx2word = sentence2dataset.make_dict(words) # (vocab, index)
    training_data = sentence2dataset.make_training_data(words, window_size)

    
    model = SkipGram(vocab=vocab_size, embedding_size=embedding_size, window_size=window_size, device=device).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    model.train()

    epochs = 500
    for epoch in tqdm(range(epochs)):
        for i, batch in enumerate(training_data):
            target, center = batch
            center = sentence2dataset.one_hot_encoding(center, word2idx)
            center = torch.tensor(center, dtype=torch.float).to(device)
            target = torch.tensor([word2idx[word] for word in target], dtype=torch.long).to(device)

            optimizer.zero_grad()
            output = model(center)

            loss = criterion(output,target)
            loss.backward()
            optimizer.step()


    # TESTING
    center = 'People'
    center_vector = sentence2dataset.one_hot_encoding(center, word2idx)
    center_vector = torch.tensor(center_vector, dtype=torch.float).to(device)
    idx2word = {}
    for word in word2idx.keys():
        idx2word[word2idx[word]]=word
    
    model.eval()
    a = model(center_vector)

    #Print result
    print('Context : ', center)
    print('Prediction : ', [idx2word[torch.argmax(r).item()] for r in a])

if __name__ == "__main__":
    main()