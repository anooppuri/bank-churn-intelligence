"""
models.py
Model definitions + Optuna hyperparameter tuning + isotonic probability calibration.
Compatible with scikit-learn >= 1.3.
"""

import os
import pickle
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def get_base_models():
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            random_state=42, solver="lbfgs"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=50, min_samples_leaf=20,
            class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=30,
            min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_split=30, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=4,
            eval_metric="logloss", random_state=42, verbosity=0)
    return models


def tune_gradient_boosting(X_train, y_train, n_trials=40):
    if not OPTUNA_AVAILABLE:
        print("  [WARN] Optuna not available. Using default GB params.")
        return GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_split=30, random_state=42), None

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 6),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 80),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 5, 40),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
        }
        model = GradientBoostingClassifier(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train,
                               cv=skf, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    return GradientBoostingClassifier(**best), study.best_value


def tune_xgboost(X_train, y_train, n_trials=40):
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        return None, None

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 6.0),
            "eval_metric": "logloss", "random_state": 42, "verbosity": 0,
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train,
                               cv=skf, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"eval_metric": "logloss", "random_state": 42, "verbosity": 0})
    return XGBClassifier(**best), study.best_value


def calibrate_model(model, X_train, y_train, method="isotonic", cv=5):
    """
    Isotonic regression calibration via cross-validation.
    Ensures predicted probability 0.70 ≈ 70% empirical churn rate.
    Compatible with sklearn >= 1.3 (cv='prefit' removed).
    """
    cal = CalibratedClassifierCV(clone(model), method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
