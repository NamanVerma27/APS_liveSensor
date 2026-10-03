import os, sys
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from sensor.exception import SensorException
from sensor.entity.artifact_entity import ClassificationMetricArtifact


def calculate_business_cost(y_true, y_pred) -> float:
    """
    Calculates the Scania APS business failure cost:
    Cost 1 (False Positive / False Alarm): $10
    Cost 2 (False Negative / Missed Failure): $500
    """
    try:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
            fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
        return float((10 * fp) + (500 * fn))
    except Exception:
        return 0.0


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculates classification metrics: F1 score, precision, recall, and Scania business cost.
    """
    try:
        model_f1_score = f1_score(y_true, y_pred, zero_division=0)
        model_recall_score = recall_score(y_true, y_pred, zero_division=0)
        model_precision_score = precision_score(y_true, y_pred, zero_division=0)
        cost = calculate_business_cost(y_true, y_pred)

        return ClassificationMetricArtifact(
            f1_score=float(model_f1_score),
            precision_score=float(model_precision_score),
            recall_score=float(model_recall_score),
            cost=cost
        )
    except Exception as e:
        raise SensorException(e, sys)