import torch


IMDB_DATA_PATH = '/data/wangchenghao/imdb'
EMBEDDING_PATH = '/data/wangchenghao/glove.txt'
UNK_STR = '<unk>'
UNK_TOKEN = 0
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_AVAILABLE else 'cpu'
MODEL_PATH = '/data/wangchenghao/sentiment_model.pkl'
