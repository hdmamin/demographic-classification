def encode(text, w2idx, nlp):
    """Map each word in a post to its index in the embedding matrix. Posts
    retain their original lengths for now.
    """
    unk = w2idx['<UNK>']
    return [w2idx.get(word.text, unk)
            for word in nlp(text, disable=['parser', 'tagger', 'ner'])]
