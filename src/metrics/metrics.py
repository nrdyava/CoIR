import numpy as np

def calculate_recall(ground_truths, retrieved_candidates, k):
    """
    Calculate recall@k for the given dataset.

    :param data: numpy array of shape (n_samples, 52)
                 where first 2 columns are key and ground truth candidate
                 and next 50 columns are top 50 retrieved candidates.
    :param k: top k retrieved candidates to consider for recall.
    :return: recall@k value
    """

    # Check if the ground truth is within the top k retrieved candidates
    correct_predictions = np.any(retrieved_candidates[:, :k] == ground_truths[:, np.newaxis], axis=1)

    # Calculate recall@k as the mean of correct predictions
    recall_at_k = np.mean(correct_predictions)

    return recall_at_k