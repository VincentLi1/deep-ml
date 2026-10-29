import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    # Your code here
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, len(data) - block_size, size=batch_size)

    return np.array([
        data[offsets[:, None] + np.arange(block_size)],
        data[offsets[:, None] + np.arange(block_size) + 1]
    ])