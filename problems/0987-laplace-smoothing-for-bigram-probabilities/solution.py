import numpy as np

def smooth_bigram_probs(N, k):
    """
    N: 2D list or array of bigram counts (V x V)
    k: float, add-k smoothing constant (k >= 0)
    Returns: 2D list of smoothed bigram probabilities (V x V).
    """
    V = len(N)
    N = np.array(N)
    N_aug = N + k

    P = N_aug / N_aug.sum(axis=1, keepdims=True)

    P = P.tolist()
    return P


    




