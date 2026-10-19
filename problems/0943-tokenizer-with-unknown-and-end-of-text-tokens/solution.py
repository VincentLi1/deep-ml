import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == 'encode':
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        ids = [vocab.get(tok, vocab['<|unk|>']) for tok in tokens if tok and not tok.isspace()] #+ [vocab['<|endoftext|>']]
        return ids
    else:
        ids = text
        inverse_vocab = {v: k for k, v in vocab.items()}
        text = ' '.join(inverse_vocab[id_] for id_ in ids)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

