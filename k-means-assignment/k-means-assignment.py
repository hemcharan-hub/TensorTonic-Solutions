import numpy as np

def k_means_assignment(points, centroids):

    points = np.asarray(points, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)

    # Ensure 2D
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)

    # Squared Euclidean distance (vectorized)
    p2 = np.sum(points**2, axis=1, keepdims=True)
    c2 = np.sum(centroids**2, axis=1)
    dist = p2 + c2 - 2 * points @ centroids.T

    # numerical stability
    dist = np.maximum(dist, 0)

    # tie-breaking: argmin gives smallest index automatically
    return np.argmin(dist, axis=1).tolist()