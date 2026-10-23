import numpy as np


def cl_metrics(R, baseline, decimals=4):
    """ACC, BWT, FWT and forgetting from a (T, T) accuracy matrix (rows = after training task i)."""
    T = len(R)
    R = np.array(R)
    b = np.asarray(baseline)
    b = b if np.ndim(b) == 0 else b[1:]
    baseline = b

    return {
        "ACC": round(float(np.mean(R[-1])), decimals),
        "BWT": round(float(np.mean(R[-1, :-1] 
            - np.diag(R)[:-1])), decimals) if T > 1 else 0.0,
        "FWT": round(float(np.mean(R[np.arange(0, T - 1), np.arange(1, T)] - baseline)), decimals) if T > 1 else 0.0,
        "forgetting": round(float(np.mean(np.max(R[:-1], axis=0)[:-1] - R[-1, :-1])), decimals) if T > 1 else 0.0
    }


def per_task_forgetting(R, decimals=4):
    """List of length T-1: best earlier accuracy minus final accuracy, per task."""
    R = np.array(R)
    return [round(float(num), decimals) for num in np.max(R[:-1], axis=0)[:-1] - R[-1, :-1]]
