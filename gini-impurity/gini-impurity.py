import numpy as np

def gini_impurity(y_left, y_right):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    def gini(y):
        if y.size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        p = counts / y.size
        return 1.0 - np.sum(p * p)

    n_left = y_left.size
    n_right = y_right.size
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    return float(
        (n_left / n_total) * gini(y_left) +
        (n_right / n_total) * gini(y_right)
    )