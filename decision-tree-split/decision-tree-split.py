import numpy as np

def decision_tree_split(X, y):

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    n, d = X.shape

    def gini(arr):
        if arr.size == 0:
            return 0.0
        _, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)

    best_gain = -np.inf
    best_feature = None
    best_threshold = None

    for f in range(d):
        values = np.unique(X[:, f])
        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2.0

        for t in thresholds:
            left_mask = X[:, f] <= t
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if y_left.size == 0 or y_right.size == 0:
                continue

            weighted = (
                (y_left.size / n) * gini(y_left) +
                (y_right.size / n) * gini(y_right)
            )

            gain = parent_gini - weighted

            if (
                gain > best_gain or
                (np.isclose(gain, best_gain) and (best_feature is None or f < best_feature)) or
                (np.isclose(gain, best_gain) and f == best_feature and t < best_threshold)
            ):
                best_gain = gain
                best_feature = f
                best_threshold = t

    return [int(best_feature), float(best_threshold)]