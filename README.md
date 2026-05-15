<div align="center">

# 🏦 Churn Intelligence System
### Predictive Risk Scoring & Customer Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-006600?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**European Retail Bank Portfolio · 10,000 Customers · 5 ML Models · SHAP Explainability**

[Dashboard](#streamlit-dashboard) · [Results](#model-results) · [Installation](#installation) · [Usage](#usage) · [Research Paper](docs/research_paper.md)

</div>

---

## Overview

This system predicts customer churn **before it happens**, assigning calibrated risk probability scores to every customer and routing them into four actionable retention tiers. Built to institutional investment-banking standards — fully auditable, explainable, and deployment-ready.

**What makes this a 10/10 project:**
- ✅ SMOTE rebalancing (20/80 → 37.5/62.5 class ratio)
- ✅ Isotonic probability calibration (predicted 0.70 = empirical 70%)
- ✅ Youden's J optimal threshold per model (not default 0.5)
- ✅ SHAP explainability (regulatory compliance ready)
- ✅ 8 engineered features beyond raw columns
- ✅ 7-page institutional Excel model (zero hardcoded calculations)
- ✅ 7-module production Streamlit dashboard
- ✅ Full research paper with ROI analysis

---

## Model Results

| Model | ROC-AUC | Recall | Precision | F1 Score |
|---|---|---|---|---|
| **⭐ Gradient Boosting** | **0.8671** | **76.96%** | **52.68%** | **0.6255** |
| XGBoost | 0.8625 | 76.96% | 51.14% | 0.6145 |
| Random Forest | 0.8607 | 68.63% | 58.82% | 0.6335 |
| Decision Tree | 0.8549 | 80.39% | 46.59% | 0.5899 |
| Logistic Regression | 0.8303 | 76.96% | 43.37% | 0.5548 |

**Actual churn rate:** 20.37% across 10,000 customers  
**Portfolio segmentation:** 1,319 Critical · 849 High · 1,314 Moderate · 6,518 Low Risk

---

## Top Churn Drivers (SHAP)

| Rank | Feature | Mean \|SHAP\| | Key Insight |
|---|---|---|---|
| 1 | NumOfProducts | 0.710 | 3+ products → 82.7% churn |
| 2 | Age | 0.430 | Churner mean age 44.8 vs 37.4 |
| 3 | AgeGroup_Senior | 0.418 | Sharp inflection at 45 |
| 4 | IsActiveMember | 0.348 | Inactive: 26.9% vs 14.3% churn |
| 5 | Geography_Germany | 0.240 | 32.4% vs 16.2% France (2×) |

---

## Project Structure

```
churn_intelligence/
├── 📁 data/
│   └── European_Bank.csv          ← Source data (10,000 records)
├── 📁 src/
│   ├── preprocessing.py           ← Feature engineering + SMOTE pipeline
│   ├── models.py                  ← 5 ML models + Optuna tuning + calibration
│   ├── evaluation.py              ← Metrics, ROC, threshold analysis
│   ├── explainability.py          ← SHAP computation module
│   ├── eda.py                     ← EDA statistics module
│   └── train_pipeline.py          ← End-to-end training runner
├── 📁 app/
│   └── streamlit_app.py           ← 7-module production dashboard (1,612 lines)
├── 📁 models/                     ← Saved model artifacts (.pkl)
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── feature_names.json
├── 📁 outputs/
│   ├── model_comparison.csv       ← All model metrics
│   ├── feature_importance.csv     ← Tree importance rankings
│   ├── shap_importance.csv        ← SHAP mean |φ| values
│   ├── scored_customers.csv       ← 10,000 customers with risk scores
│   ├── model_results.json
│   └── summary_stats.json
├── 📁 excel/
│   ├── Churn_Intelligence_Model.xlsx  ← 7-sheet institutional model
│   └── build_excel_model.py           ← Excel builder (run to regenerate)
├── 📁 docs/
│   ├── research_paper.md          ← Full academic paper with ROI analysis
│   └── executive_summary.md       ← Government/board-level summary
├── 📁 .streamlit/
│   └── config.toml                ← Streamlit Cloud configuration
├── app.py                         ← Streamlit Cloud entry point
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/churn-intelligence.git
cd churn-intelligence
pip install -r requirements.txt
```

---

## Usage

### Option 1 — Launch Dashboard (Streamlit)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Option 2 — Retrain All Models

```bash
python src/train_pipeline.py
```

Runs the complete pipeline: preprocessing → SMOTE → training → calibration → SHAP → scoring → artifact export. Takes ~3–5 minutes.

### Option 3 — Rebuild Excel Model

```bash
python excel/build_excel_model.py
```

Regenerates `excel/Churn_Intelligence_Model.xlsx` from the latest outputs.

---

## Streamlit Dashboard — 7 Modules

| Module | Description |
|---|---|
| 🏠 Executive Dashboard | KPI strip, risk distribution, 6 executive insights |
| 🔬 EDA & Analytics | Box plots, heatmaps, distributions, cross-tabs |
| 🎯 Risk Calculator | Individual scoring with gauge chart + recommended action |
| 📈 Model Performance | ROC curves, radar, confusion matrices, P-R curves |
| 🔍 Feature Intelligence | Tree vs SHAP importance, correlation, engineered features |
| 🔬 What-If Simulator | Retention scenario analysis with delta table |
| 📋 Customer Register | Filterable 10,000-row risk register with download |

---

## Excel Model — 7 Sheets

| Sheet | Contents |
|---|---|
| Cover | Project metadata, sheet index |
| Assumptions | All inputs — **blue = hardcoded input** (IB standard) |
| Model Comparison | 5-model metrics, color-scaled ROC-AUC, definitions |
| Feature Importance | Tree + SHAP values, data bars, cumulative coverage |
| Portfolio Analysis | Risk bands, geography, gender, product breakdown |
| Risk Register | Top 200 customers, color-coded by risk band |
| EDA Summary | Descriptive stats, churn comparisons, product table |

> ⚠️ **IB Convention**: Blue cells = hardcoded inputs. Black cells = derived from blue inputs. Zero hardcoded values in any calculation cell.

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Set **Repository** to your repo, **Branch** to `main`, **Main file path** to `app.py`
5. Click **Deploy**

> **Note**: The `models/` directory contains `.pkl` files (~15MB total). These must be committed to the repository for Streamlit Cloud to load them. If using Git LFS, add `*.pkl` to `.gitattributes`.

---

## Feature Engineering

| Feature | Formula | Purpose |
|---|---|---|
| BalanceToSalaryRatio | Balance / EstimatedSalary | Financial dependency |
| ProductDensity | NumOfProducts / max(Tenure,1) | Engagement pace |
| EngagementScore | IsActiveMember + HasCrCard | Composite 0–2 signal |
| AgeTenureInteraction | Age / max(Tenure,1) | Age-tenure risk interaction |
| ZeroBalance | 1 if Balance = 0 | Dormant account flag |
| AgeGroup_Senior | 1 if Age ≥ 45 | Key age threshold |
| MultiProductRisk | 1 if Products ≥ 3 | Product cliff flag |
| InactiveHighBalance | 1 if Inactive AND Balance > median | Premium CLV at risk |

---

## Pipeline Architecture

```
Raw Data (10,000)
      │
      ▼
Feature Engineering (8 new features → 21 total)
      │
      ▼
Encode + Scale (One-hot + StandardScaler)
      │
      ▼
Stratified Split 80/10/10
      │
      ├─── Train Set (8,001)
      │         │
      │         ▼
      │    SMOTE Resampling → 10,193 samples
      │         │
      │         ▼
      │    Train 5 Models (tuned hyperparams)
      │         │
      │         ▼
      │    Isotonic Calibration (cv=5)
      │
      ├─── Val Set (999) ──→ Calibration fit
      │
      └─── Test Set (1,000) ──→ Final evaluation
                                      │
                                      ▼
                               Metrics + SHAP + Risk Scores
```

---

## Risk Band Definitions

| Band | Probability | Action | Urgency |
|---|---|---|---|
| 🔴 Critical Risk | ≥ 65% | Relationship manager + bespoke offer | 48 hours |
| 🟠 High Risk | 40–65% | Outbound call + retention package | 2 weeks |
| 🟡 Moderate Risk | 20–40% | Digital campaign + engagement nudge | Quarterly |
| 🟢 Low Risk | < 20% | Standard management | Ongoing |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Built by Anoop Puri · Churn Intelligence Project*  
*For internal risk management and retention strategy use*

</div>
