import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size, itos, seed, max_tokens):
    # Your code here
    C = np.array(C)
    rng = np.random.default_rng(seed)
    vocab_size = len(C)

    def softmax(arr, axis=-1):
        shifted = arr - np.max(arr, axis=axis, keepdims=True)
        exp_arr = np.exp(shifted)
        return exp_arr / exp_arr.sum(axis=axis, keepdims=True)

    ctx = np.zeros((block_size,), dtype=np.int32)
    ans = []

    for _ in range(max_tokens):
        
        e = C[ctx].reshape(-1) # block_size, emb_dim
        h = np.tanh(e @ W1 + b1)
        logits = h @ W2 + b2
        probs = softmax(logits, -1)
        ix = rng.choice(vocab_size, p=probs)
        if ix == 0:
            break
        ans.append(itos[ix])

        ctx[:-1] = ctx[1:]
        ctx[-1] = ix

    return ''.join(ans)