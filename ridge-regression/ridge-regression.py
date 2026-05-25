import numpy as np

def ridge_regression(X, y, lam):

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    d = X.shape[1]

    XtX = X.T @ X
    Xty = X.T @ y

    # Regularization term (lambda * I)
    I = np.eye(d, dtype=np.float64)

    w = np.linalg.inv(XtX + lam * I) @ Xty

    return w.flatten().tolist()