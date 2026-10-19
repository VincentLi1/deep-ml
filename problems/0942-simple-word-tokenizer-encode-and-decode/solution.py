import re

regex = r'([,.:;?_!"()\']|--|\s)'

def encode(text, vocab):
    tokens = re.split(regex, text)
    return [vocab[tok] for tok in tokens if tok and not tok.isspace()]

def decode(ids, vocab):
    inverse_vocab = {v: k for k, v in vocab.items()}
    return re.sub('\s+([,.?!"()\'])', r'\1', ' '.join([inverse_vocab[id_] for id_ in ids]))