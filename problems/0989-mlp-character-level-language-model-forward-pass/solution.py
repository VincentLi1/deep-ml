import numpy as np

def mlp_forward_loss(X, Y, C, W1, b1, W2, b2):
    """
    Forward pass of an MLP character-level language model.
    Returns mean cross-entropy loss as a float.
    """
    def softmax(arr, axis=-1):
        arr -= np.max(arr, axis=axis, keepdims=True)
        exp_arr = np.exp(arr)
        return exp_arr / exp_arr.sum(axis=axis, keepdims=True)

    N = len(X)
    embs = C[X] # N, block_size, emb_dim
    flat = embs.reshape(N, -1) # N, block_size * emb_dim
    h = np.tanh(flat @ W1 + b1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    loss = - np.log(probs)[np.arange(N), Y]
    return loss.mean()











