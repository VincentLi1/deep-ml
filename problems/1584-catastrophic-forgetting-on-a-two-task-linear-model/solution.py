import numpy as np


def ridge_fit(X, y, lam):
    """Closed-form ridge regression weights (no intercept)."""
    return np.linalg.solve(
        X.T @ X + lam * np.eye(X.shape[1]),
        X.T @ y
    )


def sequential_forgetting(XA, yA, XB, yB, lam=0.0, decimals=4):
    """
    Fit on A, fit on B, fit on A+B jointly. Report task-A and task-B MSEs and the forgetting
    mse_A_after_B - mse_A_after_A. All values rounded to `decimals`, as Python floats.
    """
    wA = ridge_fit(XA, yA, lam)
    wB = ridge_fit(XB, yB, lam)
    wAB = ridge_fit(np.vstack([XA, XB]), np.concatenate([yA, yB], axis=0), lam)

    def mse_loss(X, y, w):
        n = len(X)
        return (1/n) * np.linalg.norm(X @ w - y) ** 2

    return {
        'mse_A_after_A': round(mse_loss(XA, yA, wA), decimals), 
        'mse_A_after_B': round(mse_loss(XA, yA, wB), decimals), 
        'mse_B_after_B': round(mse_loss(XB, yB, wB), decimals), 
        'mse_A_joint': round(mse_loss(XA, yA, wAB), decimals), 
        'mse_B_joint': round(mse_loss(XB, yB, wAB), decimals), 
        'forgetting': round(mse_loss(XA, yA, wB) - mse_loss(XA, yA, wA), decimals), 
        'joint_total': round(mse_loss(XA, yA, wAB) +  mse_loss(XB, yB, wAB), decimals)
    }
