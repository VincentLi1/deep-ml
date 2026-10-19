import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    scores_t = torch.tensor(scores, dtype=torch.float32)
    softmax_t = torch.nn.Softmax(dim=-1)(scores_t)
    return softmax_t.tolist()
