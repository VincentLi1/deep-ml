import numpy as np

def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    if max_length >= len(token_ids):
        return []

    starts = np.arange(0, len(token_ids) - max_length, stride)
    indices = starts[:, None] + np.arange(max_length)
    token_ids = np.array(token_ids)
    inputs = token_ids[indices]
    targets = token_ids[indices + 1]
    return list(zip(inputs, targets))