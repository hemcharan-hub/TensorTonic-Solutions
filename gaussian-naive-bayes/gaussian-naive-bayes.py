import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    eps = 1e-9

    classes = np.unique(y_train)
    n, d = X_train.shape

    # Store priors, means, variances
    priors = {}
    means = {}
    vars_ = {}

    for c in classes:
        X_c = X_train[y_train == c]

        priors[c] = X_c.shape[0] / n
        means[c] = np.mean(X_c, axis=0)
        vars_[c] = np.var(X_c, axis=0) + eps  # population variance + epsilon

    predictions = []

    for x in X_test:
        best_class = None
        best_log_prob = -np.inf

        for c in classes:
            mean = means[c]
            var = vars_[c]

            log_prior = np.log(priors[c])

            # Gaussian log likelihood
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) +
                ((x - mean) ** 2) / var
            )

            log_posterior = log_prior + log_likelihood

            if log_posterior > best_log_prob:
                best_log_prob = log_posterior
                best_class = c

        predictions.append(int(best_class))

    return predictions
    