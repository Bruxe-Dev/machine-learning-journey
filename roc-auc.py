import numpy as np 

def calculate_model_auc(y_pred,y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    sort_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sort_indices]

    ranks = np.arange(1,len(y_true)+1)

    pos_ranks_sum = np.sum(ranks[y_true_sorted == 1])

    auc = (
        pos_ranks_sum
        - (n_pos * (n_pos + 1)) / 2
    ) / (n_pos * n_neg)

    return auc

actual_labels = [1, 0, 1, 0]
predicted_probs = [0.90, 0.10, 0.80, 0.40]

print("AUC Score:", calculate_auc(actual_labels, predicted_probs))