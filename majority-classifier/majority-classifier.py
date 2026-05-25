import numpy as np

def majority_classifier(y_train, X_test):

    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # Handle empty test set
    n_test = X_test.shape[0] if X_test.ndim > 0 else 0
    if y_train.size == 0 or n_test == 0:
        return np.array([], dtype=int)

    # Find classes and counts
    classes, counts = np.unique(y_train, return_counts=True)

    max_count = np.max(counts)

    # Tie-breaking: first occurring in y_train
    majority_class = None
    for label in y_train:
        if np.sum(y_train == label) == max_count:
            majority_class = label
            break

    return np.full(n_test, majority_class, dtype=int)