import numpy as np

def pca_projection(X, k):

    X = np.asarray(X, dtype=np.float64)

    # Step 1: center data
    mean = np.mean(X, axis=0)
    Xc = X - mean

    n = X.shape[0]

    # Step 2: covariance matrix (sample covariance)
    cov = (Xc.T @ Xc) / (n - 1)

    # Step 3: eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Step 4: sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    top_k_vecs = eigvecs[:, idx[:k]]

    # Step 5: projection
    X_proj = Xc @ top_k_vecs

    return X_proj.tolist()