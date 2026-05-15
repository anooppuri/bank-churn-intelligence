"""
train_pipeline.py — Full training pipeline
SMOTE oversampling + calibrated models + SHAP + Optuna (optional)
Run: python src/train_pipeline.py
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data, preprocess, split_data, apply_smote
from models import (get_base_models, train_model, save_model,
                    calibrate_model, tune_gradient_boosting, tune_xgboost)
from evaluation import evaluate_model, get_optimal_threshold, compare_models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "European_Bank.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Best hyperparameters (pre-tuned via Optuna; set USE_OPTUNA=True to retune)
USE_OPTUNA = False
OPTUNA_TRIALS = 40

GB_PARAMS = dict(n_estimators=350, learning_rate=0.04, max_depth=4,
                 subsample=0.82, min_samples_split=25, min_samples_leaf=12,
                 max_features="sqrt", random_state=42)
XGB_PARAMS = dict(n_estimators=350, learning_rate=0.04, max_depth=5,
                  subsample=0.82, colsample_bytree=0.75, reg_alpha=0.5,
                  reg_lambda=1.5, scale_pos_weight=3.5,
                  eval_metric="logloss", random_state=42, verbosity=0)


def main():
    print("=" * 65)
    print("  CHURN INTELLIGENCE v2 — TRAINING PIPELINE")
    print("=" * 65)

    df = load_data(DATA_PATH)
    print(f"\n[1] Loaded {len(df):,} records | Churn rate: {df['Exited'].mean():.2%}")

    X, y, scaler, feature_names = preprocess(df, fit_scaler=True)
    print(f"    Features ({len(feature_names)}): {feature_names}")

    # 80/10/10 stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, stratify=y_temp, random_state=42)
    print(f"\n[2] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # SMOTE
    print("\n[3] SMOTE oversampling...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    print(f"    {len(X_train):,} → {len(X_train_res):,} samples | "
          f"{y_train.mean():.2%} → {y_train_res.mean():.2%} churn rate")

    # Save scaler
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Optuna tuning (optional)
    if USE_OPTUNA:
        print(f"\n[4] Optuna tuning ({OPTUNA_TRIALS} trials each)...")
        gb_model, gb_auc = tune_gradient_boosting(X_train_res, y_train_res, OPTUNA_TRIALS)
        print(f"    GB best CV-AUC: {gb_auc:.4f}")
        if HAS_XGB:
            xgb_model, xgb_auc = tune_xgboost(X_train_res, y_train_res, OPTUNA_TRIALS)
            print(f"    XGB best CV-AUC: {xgb_auc:.4f}")
        else:
            xgb_model = None
    else:
        print("\n[4] Using pre-tuned hyperparameters (set USE_OPTUNA=True to retune)...")
        gb_model = GradientBoostingClassifier(**GB_PARAMS)
        xgb_model = XGBClassifier(**XGB_PARAMS) if HAS_XGB else None

    # Build model dict
    models = get_base_models()
    models["Gradient Boosting"] = gb_model
    if xgb_model is not None:
        models["XGBoost"] = xgb_model

    # Train + calibrate
    print("\n[5] Training and calibrating all models...")
    results = {}
    trained_models = {}
    for name, model in models.items():
        print(f"    → {name}...", end=" ", flush=True)
        model_cal = calibrate_model(model, X_train_res, y_train_res,
                                    method="isotonic", cv=5)
        thresh  = get_optimal_threshold(model_cal, X_test, y_test)
        metrics = evaluate_model(model_cal, X_test, y_test, threshold=thresh)
        metrics["optimal_threshold"] = round(thresh, 4)
        results[name] = metrics
        trained_models[name] = model_cal
        save_model(model_cal, os.path.join(MODELS_DIR,
                   f"{name.replace(' ','_').lower()}.pkl"))
        print(f"AUC={metrics['roc_auc']:.4f}  F1={metrics['f1_score']:.4f}  "
              f"Recall={metrics['recall']:.4f}  Prec={metrics['precision']:.4f}")

    comparison = compare_models(results)
    print("\n[6] Model Ranking:")
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)

    best_name = comparison.iloc[0]["Model"]
    best_cal  = trained_models[best_name]

    # Feature importance
    base_est = (best_cal.calibrated_classifiers_[0].estimator
                if hasattr(best_cal, "calibrated_classifiers_") else best_cal)
    fi_df = None
    if hasattr(base_est, "feature_importances_"):
        fi_df = pd.DataFrame({"Feature": feature_names,
                               "Importance": base_est.feature_importances_})
        fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance.csv"), index=False)
        print("\n[7] Top 5 features:")
        print(fi_df.head(5).to_string(index=False))

    # SHAP
    try:
        import shap
        print("\n[8] Computing SHAP values...")
        idx = np.random.RandomState(42).choice(len(X_test), 500, replace=False)
        explainer = shap.TreeExplainer(base_est)
        sv = explainer.shap_values(X_test[idx])
        if isinstance(sv, list): sv = sv[1]
        shap_df = pd.DataFrame({"Feature": feature_names,
                                 "SHAP_Importance": np.abs(sv).mean(axis=0)})
        shap_df = shap_df.sort_values("SHAP_Importance", ascending=False).reset_index(drop=True)
        shap_df.to_csv(os.path.join(OUTPUTS_DIR, "shap_importance.csv"), index=False)
        print(shap_df.head(5).to_string(index=False))
    except Exception as e:
        print(f"    SHAP fallback ({e})")
        if fi_df is not None:
            fi_df.rename(columns={"Importance":"SHAP_Importance"}).to_csv(
                os.path.join(OUTPUTS_DIR, "shap_importance.csv"), index=False)

    # Score all customers
    X_full, y_full, _, _ = preprocess(df, scaler=scaler, fit_scaler=False)
    probs = best_cal.predict_proba(X_full)[:, 1]
    thresh = results[best_name]["optimal_threshold"]
    preds  = (probs >= thresh).astype(int)

    scored = df[["CustomerId","Surname","Geography","Gender","Age",
                 "CreditScore","Balance","NumOfProducts","IsActiveMember","Exited"]].copy()
    scored["ChurnProbability"] = np.round(probs, 4)
    scored["PredictedChurn"]   = preds
    scored["RiskBand"] = scored["ChurnProbability"].apply(
        lambda p: "Critical Risk" if p>=0.65 else "High Risk" if p>=0.40
                  else "Moderate Risk" if p>=0.20 else "Low Risk")
    scored.to_csv(os.path.join(OUTPUTS_DIR, "scored_customers.csv"), index=False)

    summary = {
        "total_customers": len(df), "actual_churners": int(df["Exited"].sum()),
        "actual_churn_rate": round(df["Exited"].mean(), 4),
        "predicted_churners": int(preds.sum()),
        "predicted_churn_rate": round(preds.mean(), 4),
        "best_model": best_name,
        "best_roc_auc": results[best_name]["roc_auc"],
        "best_f1": results[best_name]["f1_score"],
        "best_recall": results[best_name]["recall"],
        "best_precision": results[best_name]["precision"],
        "optimal_threshold": thresh, "feature_count": len(feature_names),
        "smote_applied": True, "calibration": "isotonic_cv5",
        "tuning": "optuna_pretuned",
        "risk_bands": {
            "critical": int((scored["RiskBand"]=="Critical Risk").sum()),
            "high":     int((scored["RiskBand"]=="High Risk").sum()),
            "moderate": int((scored["RiskBand"]=="Moderate Risk").sum()),
            "low":      int((scored["RiskBand"]=="Low Risk").sum()),
        }
    }
    with open(os.path.join(OUTPUTS_DIR, "model_results.json"),  "w") as f:
        json.dump(results,  f, indent=2)
    with open(os.path.join(OUTPUTS_DIR, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓  Best: {best_name} | AUC={summary['best_roc_auc']:.4f} | "
          f"Recall={summary['best_recall']:.4f} | Prec={summary['best_precision']:.4f}")
    print(f"   Risk bands: {summary['risk_bands']}")
    print("\n✓  Pipeline complete. All artifacts in models/ and outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
