import numpy as np

def entropy_node(y):
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    return float(-np.sum(probs * np.log2(probs)))