"""
explainability.py
SHAP-based feature importance and individual prediction explanations.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_shap_values(model, X_sample, feature_names):
    """Compute SHAP values using TreeExplainer for tree-based models."""
    if not SHAP_AVAILABLE:
        return None, None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, take class-1 (churn) SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature":         feature_names,
        "SHAP_Importance": mean_abs
    }).sort_values("SHAP_Importance", ascending=False).reset_index(drop=True)

    return sv, importance_df


def explain_single_prediction(model, x_single, feature_names):
    """Return SHAP values for a single customer prediction."""
    if not SHAP_AVAILABLE:
        # Fallback: use feature importances
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            return pd.DataFrame({
                "Feature": feature_names,
                "Contribution": fi
            }).sort_values("Contribution", ascending=False).reset_index(drop=True)
        return None

    explainer = shap.TreeExplainer(model)
    x = x_single.reshape(1, -1)
    sv = explainer.shap_values(x)

    if isinstance(sv, list):
        sv = sv[1]

    contributions = sv.flatten()
    result = pd.DataFrame({
        "Feature":      feature_names,
        "Contribution": contributions
    }).sort_values("Contribution", key=abs, ascending=False).reset_index(drop=True)
    return result


def get_global_shap(model, X_test, feature_names, n_samples=500):
    """Compute global SHAP summary on a subsample."""
    if not SHAP_AVAILABLE:
        return None
    idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    sv, importance_df = compute_shap_values(model, X_test[idx], feature_names)
    return sv, importance_df, idx
