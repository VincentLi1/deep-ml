import torch

def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here
    
    def cross_entropy(p_t: torch.Tensor, q_t: torch.Tensor) -> float:
        safe_q_t = torch.where(p_t == 0, 1.0, q_t)
        return -torch.sum(
            p_t * torch.log(safe_q_t)
        ).item()
    
    return cross_entropy(P, P), cross_entropy(P, Q)
