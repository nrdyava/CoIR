import numpy as np
from collections import defaultdict


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



def calculate_APs(output, ranks):
    aps_atk = defaultdict(list)
    map_atk = {}
    for sample in output:
        gt_img_ids = np.array(sample['gt-image-ids'], dtype=int)
        predictions = np.array(sample['top_1000_ret_cands'], dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = np.cumsum(ap_labels, axis=0) * ap_labels
        precisions = precisions / np.arange(1, ap_labels.shape[0] + 1)
        
        for rank in ranks:
            aps_atk[rank].append(float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank)))
            
    for rank in ranks:
        map_atk[rank] = float(np.mean(aps_atk[rank]))
        
    return map_atk