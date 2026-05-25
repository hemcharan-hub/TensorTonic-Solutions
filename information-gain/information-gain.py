import numpy as np

def information_gain(y, split_mask):

    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    y_left = y[split_mask]
    y_right = y[~split_mask]

    def entropy(arr):
        if arr.size == 0:
            return 0.0
        _, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p))

    n = y.size
    if n == 0:
        return 0.0

    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)

    n_l = y_left.size
    n_r = y_right.size

    weighted = (n_l / n) * H_left + (n_r / n) * H_right

    return float(H_parent - weighted)