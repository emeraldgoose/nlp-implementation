from fastapi import FastAPI
from pydantic import BaseModel

import torch, torch.nn
import torch.nn.functional as F

import skipgram
from make_dataset import sentence2dataset

app = FastAPI()
device = torch.device('cpu')

class Data(BaseModel):
    string: str
    center_word: str
    prediction: list
    words: list
    word2idx: dict
    idx2word: dict
    vocab_size: int
    window_size: int
    embedding_size: int


def train(training_data):
    vocab_size = Data.vocab_size
    embedding_size = Data.embedding_size
    window_size = Data.window_size
    word2idx = Data.word2idx    

    global model
    model = skipgram.SkipGram(vocab=vocab_size, embedding_size=embedding_size, window_size=window_size).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    model.train()

    epochs = 500
    for epoch in range(epochs):
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

    return 'ok'


""" ========== 여기서부터 API 작성 ========== """


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put('/string/{string}')
def send_string(string: str):
    sentence = string
    example = sentence.strip().split(' ')
    
    words = []
    
    for word in example:
        if word == None: continue
        if word not in words: words.append(word)
    
    Data.words = words
    Data.vocab_size = len(words)
    Data.window_size = 2
    Data.embedding_size = 5

    Data.word2idx, Data.idx2word = sentence2dataset.make_dict(words) # (vocab, index)    

    return {'status': 'ok'}


@app.get('/center/{center}')
def read_result(center: str):
    training_data = sentence2dataset.make_training_data(Data.words, Data.window_size)

    ret = train(training_data)
    if ret != 'ok': return {'status' : 'failed'}

    center_vector = sentence2dataset.one_hot_encoding(center, Data.word2idx)
    center_vector = torch.tensor(center_vector, dtype=torch.float).to(device)
    
    model.eval()
    a = model(center_vector)

    ret = [Data.idx2word[torch.argmax(r).item()] for r in a]
    print(ret)

    return {'center':center, 'Prediction': ret}
