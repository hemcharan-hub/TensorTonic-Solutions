import numpy as np

def k_means_centroid_update(points, assignments, k):

    points = np.asarray(points, dtype=np.float64)
    assignments = np.asarray(assignments)

    n, d = points.shape

    centroids = np.zeros((k, d), dtype=np.float64)

    for j in range(k):
        cluster_points = points[assignments == j]

        if cluster_points.shape[0] == 0:
            centroids[j] = np.zeros(d, dtype=np.float64)
        else:
            centroids[j] = np.mean(cluster_points, axis=0)

    return centroids.tolist()