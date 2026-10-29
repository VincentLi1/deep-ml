import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    # Mapping of idx: key
    idx_to_key = [k for k in stage_latencies]
    
    # Mapping of key: idx
    key_to_idx = {key: idx for idx, key in enumerate(idx_to_key)}

    arr = [stage_latencies[k] for k in idx_to_key]
    arr = np.array(arr)

    e2e = np.sum(arr, axis=0)
    stage = np.sum(arr, axis=1)

    return {
        'e2e_mean': float(np.mean(e2e)),
        'e2e_percentiles': {a: b for a, b in zip(percentiles, np.percentile(e2e, percentiles))},
        'stage_stats': {key: {
            'mean': round(np.mean(stage[idx]), 2),
            'std': round(np.std(arr[idx]), 2),
            'percentiles': {a: round(b, 2) for a, b in zip(percentiles, np.percentile(stage[idx], percentiles))}
            } for idx, key in enumerate(idx_to_key)},
        'bottleneck': idx_to_key[np.argmax(stage)],
        'stage_pct': {
            key: round(stage[idx] * 100 / np.sum(stage), 2) for idx, key in enumerate(idx_to_key)
        }
    }