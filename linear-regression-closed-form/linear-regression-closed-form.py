import numpy as np

def linear_regression_closed_form(X, y):

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    # Normal equation: w = (X^T X)^-1 X^T y
    XtX = X.T @ X
    Xty = X.T @ y

    w = np.linalg.inv(XtX) @ Xty

    return w.flatten()