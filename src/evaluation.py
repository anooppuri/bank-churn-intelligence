"""
evaluation.py
Model evaluation: metrics, ROC curves, confusion matrix, threshold analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Return full evaluation metrics dictionary."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy":       round(accuracy_score(y_test, y_pred), 4),
        "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":         round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":       round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":        round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision":  round(average_precision_score(y_test, y_prob), 4),
        "true_positive":  int(tp),
        "false_positive": int(fp),
        "true_negative":  int(tn),
        "false_negative": int(fn),
        "specificity":    round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "threshold":      threshold
    }


def get_roc_data(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return fpr, tpr, thresholds, auc


def get_optimal_threshold(model, X_test, y_test):
    """Youden's J statistic — maximises sensitivity + specificity."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return float(thresholds[optimal_idx])


def compare_models(results_dict: dict) -> pd.DataFrame:
    rows = []
    for name, metrics in results_dict.items():
        rows.append({
            "Model":         name,
            "Accuracy":      metrics["accuracy"],
            "Precision":     metrics["precision"],
            "Recall":        metrics["recall"],
            "F1 Score":      metrics["f1_score"],
            "ROC-AUC":       metrics["roc_auc"],
            "Avg Precision": metrics["avg_precision"],
            "Specificity":   metrics["specificity"]
        })
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    return df.reset_index(drop=True)


def churn_risk_band(prob: float) -> str:
    """Assign descriptive risk band to a churn probability score."""
    if prob < 0.20:
        return "Low Risk"
    elif prob < 0.40:
        return "Moderate Risk"
    elif prob < 0.65:
        return "High Risk"
    else:
        return "Critical Risk"
