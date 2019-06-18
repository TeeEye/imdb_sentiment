import torch.nn as nn


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
        x = self.embedding(x)
        outputs, _ = self.encoder(x)
        x = self.decoder(outputs[-1])
        return x
