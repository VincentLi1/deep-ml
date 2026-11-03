import numpy as np

def bigram_counts(words):
    # Your code here
    chars = np.array([c for c in "".join(words)])
    letters = np.unique(chars)
    vocab = np.concatenate((["."], letters))
    V = len(vocab)

    text = "." + ".".join(words) + "." if words else "."
    
    _, encoded = np.unique(np.array(list(text)), return_inverse=True)

    left = encoded[:-1]
    right = encoded[1:]

    # assert False, (left, right, vocab, letters, V)

    ids = left * V + right
    N = (np.bincount(ids, minlength=V * V)).reshape(V, V)

    return N.tolist()










