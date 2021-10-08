### Custom Bert For MultiClassifier

부스트캠프 KLUE Relation Extraction Task를 수행하기 위해 Bert기반 Multiclass Single label Classification Task를 수행하기 위해 Custom layer를 구성하였습니다.  
huggingface에서 제공하는 BertForSequenceClassification의 경우, 아래와 같은 형태로 불러오게 됩니다.  

```
(bert): BertModel()
(dropout): Dropout(p=0.1, inplace=False)
(classifier): Linear(in_features=768, out_features=2, bias=True)
```

binary classification의 경우 linear layer 하나로 task를 수행하는데 적합할 수 있지만 multiclass에 대해서는 적용하기 힘들 수 있습니다.  
그래서 classifier 역할을 하는 linear layer와 bert 사이에 GRU를 넣어 multiclass에 대응할 수 있도록 구성했습니다.  

```
(bert) BertModel()
(dropout): Dropout(p=0.1, inplace=False)
(GRU): GRU(768, 768, dropout=0.1, bidirectional=False)
(classifier): Linear(in_features=768, out_feature=30, bias=True)
```

huggingface의 BertForSequenceClassifier 코드를 기반으로 하여 torch.nn.GRU를 추가했습니다.  
loss function은 multi class classification에 적합한 CrossEntropyLoss()함수를 사용했습니다.  
