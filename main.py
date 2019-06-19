import os
from utils import *
from config import *
from model import SentimentNet
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from itertools import chain
import gensim
import gensim.downloader as api
# import time


def run():
    print('Loading data...')
    train_data = load_imdb(IMDB_DATA_PATH, test=False)
    # test_data = load_imdb(IMDB_DATA_PATH, test=True)

    train_tokenized = [tokenize(sentence) for sentence, _ in train_data]
    # test_tokenized = [tokenize(sentence) for sentence, _ in test_data]

    vocab = set(chain(*train_tokenized))
    vocab_size = len(vocab)
    print('Data loaded with vocabulary size: ', vocab_size)

    print('Building embedding dict...')
    api.info('glove-wiki-gigaword-100')
    word2vec = api.load("glove-wiki-gigaword-100")
    # word2vec = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False, encoding='utf-8')
    word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
    word2idx[UNK_STR] = UNK_TOKEN
    idx2word = {idx+1: word for idx, word in enumerate(vocab)}
    idx2word[UNK_TOKEN] = UNK_STR
    print('Done')

    print('Generating ready-to-train data')
    train_features = torch.tensor(pad_sentences(idx_sentences(train_tokenized, word2idx)))
    train_labels = torch.tensor([label for _, label in train_data])
    # test_features = torch.tensor(pad_sentences(idx_sentences(test_tokenized, word2idx)))
    # test_labels = torch.tensor([label for _, label in test_data])
    print('Done')

    embed_size = 100
    weight = torch.zeros(vocab_size+1, embed_size)
    for i in range(len(word2vec.index2word)):
        try:
            index = word2idx[word2vec.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(word2vec.get_vector(idx2word[index]))

    n_epochs = 5
    embed_size = 100
    hidden_size = 100
    n_layers = 2
    bidirectional = True
    batch_size = 512
    n_labels = 2
    learning_rate = 0.5

    model = SentimentNet(embed_size=embed_size, hidden_size=hidden_size,
                         n_layers=n_layers, bidirectional=bidirectional, weight=weight, n_labels=n_labels)
    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH)
    model = model.to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_set = TensorDataset(train_features, train_labels)
    # test_set = TensorDataset(test_features, test_labels)
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print('Start training...')
    for epoch in range(n_epochs):
        # start = time.time()
        # train_loss, test_less = 0, 0
        # train_acc, test_acc = 0, 0
        n, m = 0, 0
        for feature, label in train_iter:
            n += 1
            model.zero_grad()
            feature = feature.to(DEVICE)
            label = label.to(DEVICE)
            pred = model(feature)
            loss = loss_func(pred, label)
            print('Train step: %d, loss: %.3f' % (n, loss.item()))
            loss.backward()
            optimizer.step()

    torch.save(model, MODEL_PATH)


if __name__ == '__main__':
    run()
