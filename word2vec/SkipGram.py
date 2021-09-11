"""
    Reference :
        https://wikidocs.net/22660
        https://reniew.github.io/22/
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from make_dataset import sentence2dataset

class SkipGram(nn.Module):
    def __init__(self, vocab, hids, window_size):
        super().__init__()
        self.vocab = vocab
        self.window_size = window_size
        self.in_weight = nn.Linear(in_features=vocab, out_features=hids, bias=False)
        self.embedding = nn.Embedding(num_embeddings=hids, embedding_dim=vocab)

    def forward(self, center):
        
        pass

    def word_vector(self, word, word2idx):
        pass


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
    temp = sentence2dataset.make_training_data(words, window_size)
    training_data = []

    for context, center in temp: # (context, center) -> (center, context)
        training_data.append([center, context])
    
    
    model = SkipGram(vocab=vocab_size, hids=hidden_layer_size, window_size=window_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    model.train()
    
    epochs = 1
    for epoch in tqdm(range(epochs)):
        for i, batch in enumerate(training_data):
            center, targets = batch            
            
            center = torch.tensor([word2idx[center]], dtype=torch.long).to(device)
            targets = torch.tensor([word2idx[index] for index in targets]).to(device)


            optimizer.zero_grad()
            output = model(center)
            
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
            
    

    # # TESTING
    # context = ['People', 'create', 'to', 'direct']
    # context_vector = torch.tensor([word2idx[word] for word in context]).to(device)
    # model.eval()
    # a = model(context_vector)

    # #Print result
    # print(f'Context: {context}')
    # print(f'Prediction: {words[torch.argmax(a[0]).item()]}')


if __name__ == "__main__":
    main()