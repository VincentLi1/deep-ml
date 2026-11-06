import numpy as np

def train_bigram_nn(xs, ys, vocab_size, lr=10.0, num_iters=100, alpha=0.0):
    """
    Train a one-layer neural-net bigram language model with full-batch gradient descent.

    Args:
        xs: list[int] of length N - input character indices in [0, vocab_size)
        ys: list[int] of length N - target next-character indices in [0, vocab_size)
        vocab_size: int V - number of distinct characters
        lr: float - learning rate
        num_iters: int - number of gradient descent steps
        alpha: float - L2 regularization strength on mean(W**2)

    Returns:
        list[list[float]] - trained weight matrix W of shape (V, V)
    """

    def softmax(arr, axis=-1):
        arr -= np.max(arr, axis=axis, keepdims=True)
        return np.exp(arr) / np.exp(arr).sum(axis=axis, keepdims=True)

    N = len(xs)
    V = vocab_size

    X = np.zeros((N, V))
    X[np.arange(N), np.array(xs)] = 1
    Y = np.zeros((N, V))
    Y[np.arange(N), np.array(ys)] = 1

    W = np.zeros((V, V))

    for i in range(num_iters):
        Z = X @ W
        P = softmax(Z, axis=1)
 
        grad =(1 / N) * X.T @ (P - Y) + (2 * alpha / V ** 2) * W
        W -= lr * grad
    return W.tolist()






    