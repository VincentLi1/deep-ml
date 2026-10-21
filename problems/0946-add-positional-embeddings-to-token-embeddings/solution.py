import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""
    
    _, seq_len, _ = token_embeddings.shape
    # context_length, embed_dim = pos_embedding_matrix.shape
    pos_embedding_matrix = pos_embedding_matrix[:seq_len]
    return token_embeddings + pos_embedding_matrix