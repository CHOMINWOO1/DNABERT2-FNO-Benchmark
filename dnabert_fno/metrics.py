import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, auc)


def binary_metrics(labels, probability):
    labels = np.asarray(labels)
    probability = np.asarray(probability)
    predicted = (probability >= 0.5).astype(int)
    if len(labels) == 0 or not np.isfinite(probability).all():
        raise ValueError("Metrics require nonempty, finite predictions")
    two_classes = len(np.unique(labels)) == 2
    precision, recall, _ = precision_recall_curve(labels, probability) if two_classes else (None, None, None)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probability)) if two_classes else None,
        "pr_auc": float(auc(recall, precision)) if two_classes else None,
        "average_precision": float(average_precision_score(labels, probability)) if two_classes else None,
    }
