import numpy as np

def random_forest_vote(predictions):

    predictions = np.asarray(predictions)

    n_trees, n_samples = predictions.shape

    results = []

    for i in range(n_samples):
        votes = predictions[:, i]

        classes, counts = np.unique(votes, return_counts=True)
        max_count = np.max(counts)

        # tie-break: smallest label among max votes
        best_class = np.min(classes[counts == max_count])

        results.append(int(best_class))

    return results