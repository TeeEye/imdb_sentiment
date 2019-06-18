IMDB_DATA_PATH = '/data/imdb/data'
EMBEDDING_PATH = '/data/imdb/embedding.txt'
UNK_STR = '<unk>'
UNK_TOKEN = 0
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_AVAILABLE else 'cpu'
