import torch
import torch.nn as nn
from utils import *

class SentimentNet(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, bidirectional, weight, n_labels):
        super(SentimentNet, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                               num_layers=n_layers, bidirectional=bidirectional,
                               batch_first=True, dropout=0)
        self.decoder = nn.Linear(hidden_size * (2 if bidirectional else 1), n_labels)

    def forward(self, x):
        lens = torch.zeros(x.size(0), 1)
        for i in range(x.size(0)):
            j = x.size(1) - 1
            while j > 0 and x[i][j] == 0:
                j -= 1
            lens[i][0] = j
        x = self.embedding(x)
        outputs = run_rnn(self.encoder, x, lens)
        x = self.decoder(outputs[:, -1])
        return x
