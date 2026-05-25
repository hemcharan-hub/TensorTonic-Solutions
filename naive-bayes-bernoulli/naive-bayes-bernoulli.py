import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train, d = X_train.shape

    classes = np.unique(y_train)
    n_classes = len(classes)

    # log prior
    log_prior = np.log(np.array([
        np.sum(y_train == c) / n_train for c in classes
    ], dtype=np.float64))

    # theta with Laplace smoothing
    theta = np.zeros((n_classes, d), dtype=np.float64)

    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        count_1 = np.sum(X_c, axis=0)
        n_c = X_c.shape[0]
        theta[i] = (count_1 + 1) / (n_c + 2)

    log_theta = np.log(theta)
    log_1_minus_theta = np.log(1 - theta)

    n_test = X_test.shape[0]
    out = np.zeros((n_test, n_classes), dtype=np.float64)

    for i in range(n_test):
        x = X_test[i]
        for j in range(n_classes):
            out[i, j] = log_prior[j] + np.sum(
                x * log_theta[j] + (1 - x) * log_1_minus_theta[j]
            )

    return out