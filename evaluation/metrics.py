"""Anomaly detection evaluation metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(y_true: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> dict:
    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    ap = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    precision = np.sum((labels == 1) & (y_true == 1)) / (np.sum(labels == 1) + 1e-9)
    recall = np.sum((labels == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-9)
    results = {
        "auroc": round(auroc, 4),
        "avg_precision": round(ap, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")
    return results
