import os


def load_imdb(path, test=False):
    data = []
    segment = 'train' if test else 'test'
    sentiments = {'pos': 1, 'neg': 0}
    for sentiment, label in sentiments.items():
        base_path = os.path.join(path, segment, sentiment)
        files = os.listdir(base_path)
        for file in files:
            with open(os.path.join(base_path, file)) as f:
                data.append((f.read().replace('\n', ''), label))
    return data


def tokenize(sentence):
    return [tok.lower() for tok in sentence.split()]


def idx_sentences(sentences, word2idx, unk_idx=0):
    """
    将句子的词转换为 index
    这里的 sentence 是 word list
    @param sentences: sentence list
    @param word2idx: word2idx dict
    @param unk_idx: index of unknown word
    @return idxed sentences
    """
    idxed_sentences = []
    for sentence in sentences:
        idxed_sentence = []
        for word in sentence:
            if word in word2idx:
                idxed_sentence.append(word2idx[word])
            else:
                idxed_sentence.append(unk_idx)
        idxed_sentences.append(idxed_sentence)
    return idxed_sentences


def pad_sentences(sentences, max_len=500, pad_token=0):
    """
    填充句子
    这里的 sentence 是 index list
    @param sentences: sentence list
    @param max_len: 截断长度
    @param pad_token: 填充的 index
    @return padded sentences
    """
    padded_sentences = []
    for sentence in sentences:
        padded_sentence = list(sentence[:max_len])
        while len(padded_sentence) < 500:
            padded_sentence.append(pad_token)
        padded_sentences.append(padded_sentence)
    return padded_sentences
