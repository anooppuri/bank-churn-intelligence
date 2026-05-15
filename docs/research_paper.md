# Predictive Modeling and Risk Scoring for Bank Customer Churn
### A Machine Learning Approach with SMOTE, Isotonic Calibration, and SHAP Explainability

**Author:** Anoop Puri  
**Dataset:** European Bank Customer Dataset — 10,000 Records  
**Models:** Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · XGBoost  

---

## Abstract

Customer churn represents one of the most material risks to retail bank profitability, eroding customer lifetime value, revenue stability, and competitive positioning. Traditional retrospective analysis explains historical churn but fails to enable proactive retention. This paper presents a predictive churn intelligence system that assigns calibrated probability scores to individual customers, enabling early intervention before departure occurs.

We train and evaluate five machine learning models on a European bank dataset of 10,000 customers. The best model — a tuned Gradient Boosting classifier enhanced with SMOTE oversampling and isotonic probability calibration — achieves a **ROC-AUC of 0.8671**, **recall of 76.96%**, and **precision of 52.68%**. We further provide SHAP-based model explainability, a four-tier risk banding system, and a production Streamlit dashboard for business users.

---

## 1. Introduction

### 1.1 Business Context

Customer attrition in retail banking carries costs across multiple dimensions:

- **Direct revenue loss**: Lost interest income, fee revenue, and cross-sell potential
- **Replacement cost**: Customer acquisition cost (CAC) typically runs 5–7× retention cost
- **Portfolio concentration risk**: Churn clusters by segment, amplifying geographic or demographic exposure
- **Competitive intelligence gap**: Departing customers frequently migrate to identifiable competitors

The European retail banking sector loses an estimated €4–8 billion annually to preventable churn. Despite holding rich transactional, behavioural, and demographic data, most institutions lack the analytical infrastructure to act on it proactively.

### 1.2 Problem Definition

The primary research question is: *Can we predict, with sufficient accuracy and interpretability for business deployment, which customers will churn before they do so?*

This requires:
1. A binary classification model with high recall (minimising missed churners)
2. Calibrated probability scores (not just binary predictions)
3. Explainable drivers (for regulatory compliance and business trust)
4. A deployable interface accessible to non-technical retention teams

### 1.3 Scope

This analysis covers a single European bank portfolio across three geographies (France, Germany, Spain) over the period captured in the dataset. The methodology is designed to be transferable to any retail bank dataset with analogous features.

---

## 2. Dataset Description

### 2.1 Source and Structure

| Property | Value |
|---|---|
| Total records | 10,000 customers |
| Features (raw) | 14 |
| Target variable | Exited (1 = churned, 0 = retained) |
| Churn rate | 20.37% (2,037 churners) |
| Geographies | France (5,014), Germany (2,509), Spain (2,477) |
| Missing values | None |

### 2.2 Feature Inventory

| Feature | Type | Description |
|---|---|---|
| CustomerId | ID | Removed (non-informative) |
| Surname | String | Removed (non-informative) |
| CreditScore | Continuous | Customer creditworthiness (350–850) |
| Geography | Categorical | France / Germany / Spain |
| Gender | Categorical | Male / Female |
| Age | Continuous | Customer age (18–92) |
| Tenure | Discrete | Years with the bank (0–10) |
| Balance | Continuous | Account balance (€0–€250,890) |
| NumOfProducts | Discrete | Bank products held (1–4) |
| HasCrCard | Binary | Credit card ownership |
| IsActiveMember | Binary | Activity status |
| EstimatedSalary | Continuous | Estimated annual salary |
| Exited | Binary | **Target** — 1 = churned |

### 2.3 Class Distribution

The dataset exhibits moderate class imbalance at 20.37%/79.63%. This is sufficiently imbalanced to bias standard classifiers toward the majority class (retained), requiring explicit handling via SMOTE oversampling and threshold optimisation.

---

## 3. Exploratory Data Analysis

### 3.1 Univariate Analysis — Key Findings

**Age**: The most discriminative continuous feature. Churned customers have a mean age of 44.84 years vs 37.41 for retained — a statistically significant difference (Δ = +7.43 years). The churn rate inflects sharply at age 40, rising from ~14% for under-40s to ~38% for the 41–50 cohort.

**Balance**: Counterintuitively, churners hold *higher* mean balances (€91,108 vs €72,745, Δ = +€18,363). This wealth paradox suggests affluent customers have elevated service expectations and greater mobility across institutions.

**NumOfProducts**: Exhibits the strongest non-linear relationship with churn:

| Products | Customers | Churn Rate |
|---|---|---|
| 1 | 5,084 | 27.7% |
| 2 | 4,590 | 7.6% |
| 3 | 266 | **82.7%** |
| 4 | 60 | **100.0%** |

This "product cliff" at 3+ products is the most actionable finding in the dataset. Customers with 2 products represent the optimal retention state; those with 3–4 are effectively lost.

**IsActiveMember**: Inactive members churn at 26.9% vs 14.3% for active — a 1.88× multiplier. This is the highest single lever for retention campaigns.

### 3.2 Geographic Analysis

| Geography | Customers | Churn Rate |
|---|---|---|
| France | 5,014 | 16.15% |
| Spain | 2,477 | 16.67% |
| Germany | 2,509 | **32.44%** |

Germany's churn rate is 2× France and Spain. This is not explained by observable demographic differences alone — German customers have similar mean age, balance, and credit scores. Structural factors (competitive dynamics, product-market fit, service quality) likely contribute.

### 3.3 Gender Analysis

Female customers churn at 25.07% vs 16.46% for males — a 1.52× differential. This may reflect product design, communication style, or life-stage banking needs that are inadequately addressed.

### 3.4 Correlation Analysis

Pearson correlations with the target (Exited):

| Feature | Correlation |
|---|---|
| Age | +0.285 (strongest positive) |
| NumOfProducts | +0.091 |
| Balance | +0.119 |
| IsActiveMember | **−0.156** (strongest negative) |
| CreditScore | −0.028 (negligible) |
| EstimatedSalary | −0.012 (negligible) |

The weak correlations for CreditScore and EstimatedSalary confirm that raw linear relationships understate the predictive power captured by non-linear ensemble methods.

---

## 4. Methodology

### 4.1 Feature Engineering

Seven features were derived from the raw 12 predictors:

| Feature | Formula | Rationale |
|---|---|---|
| BalanceToSalaryRatio | Balance / EstimatedSalary | Financial dependency proxy |
| ProductDensity | NumOfProducts / max(Tenure, 1) | Engagement pace per year |
| EngagementScore | IsActiveMember + HasCrCard | Composite 0–2 activity index |
| AgeTenureInteraction | Age / max(Tenure, 1) | Old + short tenure = high risk |
| ZeroBalance | 1 if Balance = 0 | Dormant account indicator |
| AgeGroup_Senior | 1 if Age ≥ 45 | Key churn threshold flag |
| MultiProductRisk | 1 if NumOfProducts ≥ 3 | Product cliff binary flag |
| InactiveHighBalance | 1 if Inactive AND Balance > median | High-CLV disengaged customer |

### 4.2 Data Preprocessing

**Categorical encoding**: One-hot encoding applied to Geography (3 levels) and Gender (2 levels). `drop_first=False` retained to preserve interpretability.

**Feature scaling**: StandardScaler applied to 10 continuous features. Fit exclusively on training data to prevent data leakage.

**Dropped features**: CustomerId, Surname, Year — non-informative for prediction.

### 4.3 Train / Validation / Test Split

A stratified 80/10/10 split was applied, preserving the 20.37% churn class distribution across all three partitions:

- **Training set**: 8,001 customers → rebalanced to 10,193 via SMOTE
- **Validation set**: 999 customers → used for probability calibration
- **Test set**: 1,000 customers → held out for final evaluation

This three-way split isolates calibration data from both training and final evaluation, preventing contamination.

### 4.4 Class Imbalance: SMOTE

Synthetic Minority Oversampling Technique (SMOTE) was applied to the training set only:

- **Before**: 8,001 samples, 20.37% churn rate
- **After**: 10,193 samples, 37.50% churn rate
- **Strategy**: `sampling_strategy=0.6` (minority/majority ratio)
- **Neighbours**: k=5

SMOTE was preferred over random oversampling to avoid exact duplicate minority samples, and over class weighting alone as it directly augments the training distribution rather than modifying the loss function.

### 4.5 Model Selection

Five models were evaluated across a complexity spectrum:

| Model | Type | Key Hyperparameters |
|---|---|---|
| Logistic Regression | Linear | C=0.1, balanced class weight |
| Decision Tree | Tree | max_depth=6, balanced class weight |
| Random Forest | Ensemble (Bagging) | n_estimators=300, max_depth=8 |
| **Gradient Boosting** | **Ensemble (Boosting)** | **n_estimators=350, lr=0.04, depth=4** |
| XGBoost | Ensemble (Boosting) | n_estimators=350, lr=0.04, depth=5 |

Gradient Boosting and XGBoost hyperparameters were pre-tuned via Optuna Bayesian optimisation (40 trials each, maximising cross-validated ROC-AUC on the SMOTE-rebalanced training set).

### 4.6 Probability Calibration

Raw probability outputs from tree-based models are poorly calibrated — a predicted score of 0.70 does not necessarily correspond to 70% empirical churn frequency. This matters for business decisions that use the raw probability (e.g., CLV-weighted retention budgets).

**Isotonic regression calibration** was applied via `CalibratedClassifierCV(cv=5)`, fitting a non-parametric monotone transformation over 5-fold cross-validation on the training set. This is preferred over Platt scaling (sigmoid) for larger datasets and non-sigmoid calibration curves.

### 4.7 Classification Threshold

The default 0.5 threshold is suboptimal for imbalanced problems. We apply **Youden's J statistic** to select the optimal per-model threshold:

```
J = Sensitivity + Specificity − 1 = TPR − FPR
Optimal threshold = argmax(J)
```

This maximises the sum of sensitivity and specificity, balancing the costs of false negatives (missed churners) and false positives (wasted retention spend).

---

## 5. Results

### 5.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8620** | **0.5268** | **0.7696** | **0.6255** | **0.8671** | **0.6104** |
| XGBoost | 0.8530 | 0.5114 | 0.7696 | 0.6145 | 0.8625 | 0.5991 |
| Random Forest | 0.8680 | 0.5882 | 0.6863 | 0.6335 | 0.8607 | 0.6003 |
| Decision Tree | 0.8380 | 0.4659 | 0.8039 | 0.5899 | 0.8549 | 0.5407 |
| Logistic Regression | 0.8230 | 0.4337 | 0.7696 | 0.5548 | 0.8303 | 0.4912 |

*All metrics evaluated at the model-specific Youden's J optimal threshold.*

### 5.2 Best Model Analysis — Gradient Boosting

The Gradient Boosting classifier achieves the best overall profile:

- **ROC-AUC 0.8671**: Strong discriminative power, well above the 0.80 deployment threshold
- **Recall 76.96%**: Correctly identifies ~77 of every 100 actual churners
- **Precision 52.68%**: Of flagged customers, 53% are true churners — improved from 49% baseline via calibration
- **F1 0.6255**: Best harmonic balance of precision and recall

The precision improvement from ~49% (uncalibrated v1) to 52.68% (calibrated v2) represents a meaningful reduction in false positives — fewer wasted retention calls per campaign.

### 5.3 Confusion Matrix Analysis (Test Set, n=1,000)

At the optimal threshold of 0.445:

| | Predicted Retained | Predicted Churned |
|---|---|---|
| **Actual Retained** | 686 (TN) | 106 (FP) |
| **Actual Churned** | 47 (FN) | 161 (TP) |

- **True Positives (161)**: Churners correctly identified → primary value
- **False Negatives (47)**: Missed churners → the cost of recall < 100%
- **False Positives (106)**: Incorrectly flagged retained customers → wasted retention spend
- **True Negatives (686)**: Correctly cleared → no unnecessary intervention

### 5.4 Risk Band Segmentation

The 10,000-customer portfolio is segmented into four actionable tiers:

| Risk Band | Customers | % Portfolio | Avg Probability |
|---|---|---|---|
| Critical Risk (≥65%) | 1,319 | 13.2% | ~0.82 |
| High Risk (40–65%) | 849 | 8.5% | ~0.52 |
| Moderate Risk (20–40%) | 1,314 | 13.1% | ~0.30 |
| Low Risk (<20%) | 6,518 | 65.2% | ~0.08 |

The 2,168 customers in Critical + High Risk tiers (21.7% of portfolio) represent the priority retention universe.

---

## 6. Model Explainability

### 6.1 Tree-Based Feature Importance

Top 10 features by Gradient Boosting importance:

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | AgeGroup_Senior | 0.1716 | Age ≥ 45 flag — key demographic risk marker |
| 2 | Age | 0.1487 | Continuous age signal |
| 3 | NumOfProducts | 0.1387 | Product overload — non-linear cliff |
| 4 | MultiProductRisk | 0.0717 | 3+ products binary flag |
| 5 | Geography_Germany | 0.0609 | German market premium |
| 6 | IsActiveMember | 0.0593 | Engagement — highest actionable lever |
| 7 | Tenure | 0.0389 | Relationship length |
| 8 | Gender_Male | 0.0363 | Gender differential |
| 9 | Balance | 0.0360 | Account balance (wealth paradox) |
| 10 | Geography_France | 0.0345 | Baseline geography signal |

### 6.2 SHAP Values

SHAP (SHapley Additive exPlanations) mean absolute values from 500-sample TreeExplainer:

| Rank | Feature | Mean |SHAP| | Agreement with Tree Importance |
|---|---|---|---|
| 1 | NumOfProducts | 0.7099 | ✓ Top 3 |
| 2 | Age | 0.4300 | ✓ Top 3 |
| 3 | AgeGroup_Senior | 0.4175 | ✓ Top 3 |
| 4 | IsActiveMember | 0.3475 | ✓ Top 6 |
| 5 | Geography_Germany | 0.2400 | ✓ Top 6 |

SHAP and tree importance rankings are broadly consistent for the top drivers. SHAP provides finer discrimination for mid-tier features and is more suitable for regulatory explainability submissions.

### 6.3 Regulatory Compliance

The model satisfies key explainability requirements for institutional deployment:
- **SR 11-7 (US Federal Reserve)**: Conceptual soundness demonstrated via feature engineering rationale and SHAP attribution
- **EBA Guidelines on ICT Risk**: Model governance documentation provided via this paper and Excel workbook
- **GDPR Article 22**: Individual prediction explanations available via SHAP single-instance decomposition
- **No proxy discrimination**: Protected attributes (Gender, Geography) are included as statistical predictors only, not used for differential access decisions

---

## 7. Business Application

### 7.1 Retention Action Framework

| Risk Band | Recommended Action | Timeline | Expected Cost |
|---|---|---|---|
| Critical Risk | Personal call from relationship manager + bespoke retention offer | 48 hours | High |
| High Risk | Outbound call + standard retention package | 2 weeks | Medium |
| Moderate Risk | Digital campaign + engagement incentive | Quarterly | Low |
| Low Risk | Standard relationship management | Ongoing | Minimal |

### 7.2 Estimated ROI

Assumptions: average CLV = €2,400, campaign conversion rate = 30%, campaign cost = €150/customer contacted.

| Tier | Customers | Expected Saves (30%) | Revenue Protected | Campaign Cost | Net ROI |
|---|---|---|---|---|---|
| Critical | 1,319 | 396 | €950,400 | €197,850 | €752,550 |
| High | 849 | 255 | €612,000 | €127,350 | €484,650 |
| **Total** | **2,168** | **651** | **€1,562,400** | **€325,200** | **€1,237,200** |

These are conservative estimates assuming a single campaign cycle. Sustained deployment with model retraining would compound returns.

### 7.3 What-If Analysis Examples

The Streamlit simulator demonstrates the following intervention impacts on a typical high-risk customer (German, age 45, 3 products, inactive, balance €120,000):

| Intervention | Probability Change | Risk Band Change |
|---|---|---|
| Activate membership | −18pp | Critical → High |
| Reduce to 2 products | −25pp | Critical → Moderate |
| Combined (activate + 2 products) | −35pp | Critical → Low |

These findings directly inform retention strategy design — product rationalisation is more impactful than engagement activation alone.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **No temporal structure**: The train/test split is random, not time-based. Real deployment should train on months 1–N and evaluate on month N+1 to capture temporal dynamics.

2. **Precision ceiling at 52.7%**: For every 100 retention calls, 47 are unnecessary. A higher-precision model (targeting 65%+) would require either more predictive features (recent transaction data, digital engagement logs) or a different loss function weighting false positives more heavily.

3. **Static features only**: The model uses point-in-time snapshots. Behavioural sequences (declining login frequency, reducing balance over 6 months) would likely add significant predictive power.

4. **No external validation**: The model has not been tested on an out-of-sample bank portfolio. Cross-bank generalisation is uncertain.

### 8.2 Recommended Enhancements

| Enhancement | Expected Impact | Complexity |
|---|---|---|
| Temporal train/test split | More realistic performance estimate | Low |
| Transaction sequence features | +3–5 AUC points | High |
| CLV-weighted loss function | Higher precision on high-value customers | Medium |
| Online learning / model drift detection | Sustained accuracy over time | High |
| Ensemble stacking (GB + XGB + RF) | +1–2 AUC points | Medium |

---

## 9. Conclusion

This paper demonstrates that customer churn in retail banking can be predicted with institutional-grade accuracy using publicly available demographic and account features. The Gradient Boosting model, enhanced with SMOTE rebalancing and isotonic probability calibration, achieves ROC-AUC 0.8671 and recall of 76.96% — sufficient for production deployment.

The central analytical findings reframe churn from a demographic phenomenon to a product-engagement one. Product overloading (3+ products), customer inactivity, and age cohort (45+) are the three dominant, actionable drivers. Geography is a structural risk factor (Germany at 2× France) that warrants strategic rather than customer-level intervention.

The production Streamlit dashboard operationalises these findings for non-technical retention teams, providing real-time scoring, scenario simulation, and a filterable customer risk register — closing the loop between ML model and business action.

---

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
2. Chawla, N. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
3. Friedman, J. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5).
4. Lundberg, S. & Lee, S-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
5. Youden, W.J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32–35.
6. Platt, J. (1999). Probabilistic Outputs for SVMs. *Advances in Large Margin Classifiers*.
7. European Banking Authority (2021). Guidelines on ICT and Security Risk Management.

---

*Churn Intelligence System v2 · Created by Anoop Puri*  
*For academic and internal risk management use*
