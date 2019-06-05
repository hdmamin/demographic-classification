import spacy
from torch.utils.data import Dataset, DataLoader

from utils import load_pickle, save_pickle


class BlogDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y.values

    def __getitem__(self, i):
        return encode(self.x[i], w2idx, nlp), self.y[i]

    def __len__(self):
        return self.x.shape[0]


def encode(text, w2idx, nlp):
    """Map each word in a post to its index in the embedding matrix. Posts
    retain their original lengths for now.
    """
    unk = w2idx['<UNK>']
    return [w2idx.get(word.text, unk)
            for word in nlp(text, disable=['parser', 'tagger', 'ner'])]


w2idx = load_pickle('w2idx')
nlp = spacy.load('en_core_web_sm')
