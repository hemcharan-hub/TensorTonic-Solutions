import numpy as np

def knn_distance(X_train, X_test, k):

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Ensure 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if n_train == 0:
        return np.full((n_test, k), -1, dtype=int)

    # Euclidean distance (stable)
    diff = X_test[:, None, :] - X_train[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Stable sorting for tie handling
    idx = np.argsort(dist, axis=1, kind="mergesort")

    # Handle k > n_train
    if k <= n_train:
        return idx[:, :k].astype(int)

    out = np.full((n_test, k), -1, dtype=int)
    out[:, :n_train] = idx
    return out