from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def calculate_classification_metrics(true_labels: List[int], predicted_labels: List[int],
                                     y_scores: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculates various classification metrics for binary classification.

    Parameters:
    - y_true: List[int], true labels
    - y_pred: List[int], predicted labels
    - y_scores: List[float], predicted scores or probabilities for the positive class

    Returns:
    - Dict[str, float], dictionary with keys as metric names and values as metric scores
    """
    metrics = {'accuracy': accuracy_score(true_labels, predicted_labels),
               'precision': precision_score(true_labels, predicted_labels),
               'recall': recall_score(true_labels, predicted_labels),
               'f1_score': f1_score(true_labels, predicted_labels),
               'roc_auc': roc_auc_score(true_labels, y_scores) if y_scores is not None else None}
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    metrics.update({
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    })

    return metrics
