import torch
from utils import *
from config import *


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
        lens = []
        for i in range(x.size(0)):
            j = x.size(1) - 1
            while j > 0 and x[i][j] == 0:
                j -= 1
            lens.append(j)
        lens = torch.tensor(lens).long().to(DEVICE)
        x = self.embedding(x)
        outputs = run_rnn(self.encoder, x, lens)
        x = self.decoder(outputs[:, -1])
        return x
