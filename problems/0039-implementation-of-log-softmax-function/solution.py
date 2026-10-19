import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    scores_t = torch.tensor(scores, dtype=torch.float32)
    result_t = torch.nn.LogSoftmax(dim=0)(scores_t)
    return result_t

