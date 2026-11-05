import numpy as np

def bigram_avg_nll(P, words, stoi):
    """Average negative log-likelihood of words under a bigram model."""
    if not words: return 0.0

    text = "".join("." + w for w in words) + "."

    pairs = [(stoi[text[i]], stoi[text[i+1]]) for i in range(len(text) - 1)]
    if not pairs: return 0.0

    pairs = np.array(pairs)
    probs = P[pairs[:, 0], pairs[:, 1]]
    return float(round( -np.log(probs).mean(), 4))





    



    

    
