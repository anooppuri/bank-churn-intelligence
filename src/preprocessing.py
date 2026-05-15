"""
preprocessing.py
Data preprocessing, feature engineering, SMOTE oversampling pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

DROP_COLS  = ["CustomerId", "Surname", "Year"]
CAT_COLS   = ["Geography", "Gender"]
SCALE_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "BalanceToSalaryRatio", "ProductDensity",
    "EngagementScore", "AgeTenureInteraction"
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BalanceToSalaryRatio"] = np.where(
        out["EstimatedSalary"] > 0, out["Balance"] / out["EstimatedSalary"], 0)
    out["ProductDensity"]       = out["NumOfProducts"] / out["Tenure"].clip(lower=1)
    out["EngagementScore"]      = out["IsActiveMember"] + out["HasCrCard"]
    out["AgeTenureInteraction"] = out["Age"] / out["Tenure"].clip(lower=1)
    out["ZeroBalance"]          = (out["Balance"] == 0).astype(int)
    out["AgeGroup_Senior"]      = (out["Age"] >= 45).astype(int)
    out["MultiProductRisk"]     = (out["NumOfProducts"] >= 3).astype(int)
    out["InactiveHighBalance"]  = (
        (out["IsActiveMember"] == 0) &
        (out["Balance"] > out["Balance"].median())
    ).astype(int)
    return out


def preprocess(df: pd.DataFrame, scaler=None, fit_scaler: bool = True):
    df = engineer_features(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    target = "Exited"
    y = df[target].values
    X = df.drop(columns=[target])

    X = pd.get_dummies(X, columns=CAT_COLS, drop_first=False)
    feature_names = X.columns.tolist()

    cols_to_scale = [c for c in SCALE_COLS if c in X.columns]
    if fit_scaler:
        scaler = StandardScaler()
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    else:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X.values, y, scaler, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def apply_smote(X_train, y_train, random_state=42):
    """SMOTE oversampling — corrects 20/80 class imbalance before training."""
    if not SMOTE_AVAILABLE:
        print("  [WARN] imbalanced-learn not available. Skipping SMOTE.")
        return X_train, y_train
    sm = SMOTE(sampling_strategy=0.6, random_state=random_state, k_neighbors=5)
    return sm.fit_resample(X_train, y_train)
