import numpy as np

def pad_and_truncate(sequences, pad_token_id, max_length=None):
    """
    Pad and/or truncate tokenized sequences to a uniform length.

    Args:
        sequences: list of lists of int token ids
        pad_token_id: int, token id used for padding
        max_length: int or None, target length. If None, use the longest sequence length.

    Returns:
        list of lists of int token ids, each of length max_length.
    """
    if sequences == []: return []
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    for i in range(len(sequences)):
        if len(sequences[i]) > max_length:
            sequences[i] = sequences[i][:max_length]
        else:
            sequences[i] += [pad_token_id] * (max_length - len(sequences[i]))
    return sequences
    



    return []
