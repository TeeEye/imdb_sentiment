import torch
from torchtext import data
import random
from torchtext import datasets
import torch.nn as nn
import torch.optim as optim


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))


SEED = 1234

torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)


train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print('#Train: %d #Test: %d', len(train_data), len(test_data))


train_data, valid_data = train_data.split(random_state=random.seed(SEED))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.100d',
                 unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print('Most common words: ', TEXT.vocab.freqs.most_common(20))

CUDA_AVALABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVALABLE else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=DEVICE
)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

optimizer = optim.SGD(model.parameters(), lr=1e-3)

crit = nn.BCEWithLogitsLoss()

model = model.to(DEVICE)
crit = crit.to(DEVICE)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / correct.size(0)


def train(model, iterator, optimizer, crit):
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pred = model(batch.text).squeeze(1)
        loss = crit(pred, batch.label)
        loss.backward()
        optimizer.step()
        acc = binary_accuracy(pred, batch.label)
        print('Loss: %f Acc: %f' % (loss.item(), acc.item()))


N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train(model, train_iterator, optimizer, crit)
