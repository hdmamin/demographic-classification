from collections import Counter
import numpy as np
import pandas as pd
import spacy

from utils import train_val_test_split, save_pickle, load_glove


# Must disable parser, tagger, and ner in nlp() when working with the whole
# dataset to avoid memory issues.
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 600_000_000
glove_dir = '/Users/hmamin/data/glove/'


def build_word_mappings(x_train, nlp, glove_dir):
    """Generate word to count, word to index, and word to vector mappings."""
    # Map each token to the # of times it appears in the corpus.
    tokens = [item for t in nlp(' '.join(x_train.values),
                                disable=['parser', 'tagger', 'ner'])
              for item in [t.text.strip()] if item]
    w2count = dict(filter(lambda x: x[1] > 4, Counter(tokens).items()))
    save_pickle(tokens, 'tokens')
    save_pickle(w2count, 'w2count')

    # Construct w2idx dict and i2w list.
    w2idx = {k: i for i, (k, v) in enumerate(sorted(w2count.items(),
                                                    key=lambda x: x[1],
                                                    reverse=True), 2)}
    w2idx['<PAD>'] = 0
    w2idx['<UNK>'] = 1
    i2w = [k for k, v in sorted(w2idx.items(), key=lambda x: x[1])]
    save_pickle(w2idx, 'w2idx')
    save_pickle(i2w, 'i2w')

    # Load word vectors and filter to include words in our vocab.
    w2vec = load_glove(300, glove_dir)
    w2vec = {k: v for k, v in w2vec.items() if k in w2idx}
    save_pickle(w2vec, 'w2vec')


def main(nlp, glove_dir):
    """Filter out sentences that are too short to be meaningful or far longer
    than the rest of our data.

    Parameters
    -----------
    nlp: spacy.lang.en.English
        Spacy parser used for tokenization.
    glove_dir: str
        Location to load glove vectors from.
    """
    # Load and split data.
    dtypes = dict(posts=object, sex='category', age=np.int8)
    df = pd.read_csv('data/posts.csv', dtype=dtypes, usecols=dtypes.keys())
    df['sex'] = (df.sex == 'male') * 1
    lengths = df.posts.str.split().str.len()
    df = df[(lengths >= 5) & (lengths <= 50)]
    data = train_val_test_split(df.posts, df[['sex', 'age']], train_p=.96,
                                val_p=.02, state=1, shuffle=True)
    # Order: x_train, x_val, x_test, y_train, y_val, y_test
    save_pickle(data, 'split_data')

    # w2count, w2idx, i2w, and w2vec will be pickled for easy access.
    build_word_mappings(data[0], nlp, glove_dir)


if __name__ == '__main__':
    main(nlp, glove_dir)
