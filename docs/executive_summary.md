# Executive Summary
## Churn Intelligence System — European Retail Bank
### Prepared for Government & Institutional Stakeholders

---

## Background

Customer churn costs the European retail banking sector an estimated €4–8 billion annually in lost lifetime value and replacement acquisition costs. Despite holding rich transactional and behavioral data, most banks operate churn detection reactively — flagging departures after they occur rather than preventing them.

This project delivers a **predictive churn intelligence system** that assigns real-time risk scores to customers, enabling retention strategy to shift from reactive to proactive.

---

## What We Built

A machine learning system trained on 10,000 European bank customers that:

1. **Predicts churn probability** for every customer on a 0–100% scale
2. **Classifies customers** into four risk tiers with associated retention actions
3. **Identifies the root causes** of churn through explainable AI methods
4. **Simulates intervention outcomes** so retention teams can test strategies before deploying

The best model (Gradient Boosting) achieves **ROC-AUC of 0.8690**, correctly identifying **79.1% of customers who will churn** before they leave.

---

## Key Findings

### Finding 1: Germany Is the Highest-Risk Market
German customers churn at **32.4%**, more than double the French rate of 16.2%. This 2× differential points to product-market fit issues, service quality gaps, or competitive dynamics specific to the DACH region. Retention investment should be concentrated here first.

### Finding 2: Product Overloading Destroys Loyalty
Customers with 3 bank products churn at **82.7%**; those with 4 products churn at **100%**. The intuitive assumption that more products equals more loyalty is inverted in this data. Customers forced into product bundles they don't want or use are the most likely to leave. The optimal product count for retention is 2.

### Finding 3: Engagement Is the Primary Lever
Inactive members are more than twice as likely to churn as active ones. This is the highest-impact retention lever because it is directly actionable — the bank can drive engagement through targeted communications, product activation campaigns, and digital onboarding improvements.

### Finding 4: Age 45+ Is the High-Value, High-Risk Cohort
The average churned customer is **44.8 years old** vs 37.4 for retained customers. This cohort typically holds higher balances and salary levels, meaning each lost customer in this segment represents disproportionate CLV destruction. Age-targeted retention programs are justified.

### Finding 5: Balance Doesn't Protect Against Churn
Contrary to intuition, customers with **higher balances churn more** (mean €91,108 for churners vs €72,745 for retained). This suggests affluent customers have higher service expectations and are more mobile across institutions. Premium service tiers for high-balance customers should be a priority.

---

## Risk Segmentation: Portfolio Snapshot

| Risk Band | Customers | Actual Churn Rate | Recommended Action |
|---|---|---|---|
| Critical Risk (≥65%) | 444 | ~79% | Immediate 1:1 retention outreach |
| High Risk (40–65%) | 756 | ~56% | Proactive campaign within 2 weeks |
| Moderate Risk (20–40%) | 1,842 | ~28% | Quarterly engagement touchpoint |
| Low Risk (<20%) | 6,958 | ~8% | Standard relationship management |

The 1,200 customers in Critical + High Risk tiers represent the priority retention universe — roughly 12% of the portfolio, but responsible for a disproportionate share of churn events.

---

## Regulatory & Compliance Considerations

1. **Model explainability**: All predictions are backed by SHAP value analysis and feature importance rankings, satisfying MiFID II and EBA guidelines on model governance and auditability.

2. **No demographic discrimination**: Gender and geography are included as statistical predictors, not as basis for differential service. The model is used to *improve* service delivery, not restrict access.

3. **Data minimisation**: The model uses only data already held by the bank in normal course of business. No additional personal data collection is required.

4. **Model validation**: The 80/20 stratified split and cross-validation methodology ensure results are not overfit and will generalise to new customer populations.

---

## Estimated Business Impact

Assuming a retention campaign converts 30% of high-risk customers, and average CLV of €2,400 per customer:

| Scenario | Customers Saved | Revenue Protected |
|---|---|---|
| Convert 30% of Critical Risk | ~133 | ~€320,000 |
| Convert 30% of High Risk | ~227 | ~€545,000 |
| Combined (top 1,200) | ~360 | ~€864,000 |

These are conservative estimates. Actual CLV in a full bank portfolio would be materially higher.

---

## Recommendations

1. **Deploy the Streamlit dashboard** to relationship managers and retention teams immediately — no ML expertise required to use the Risk Calculator and What-If Simulator.

2. **Prioritise the 444 Critical Risk customers** for personal outreach within 48 hours. This cohort has the highest probability and the most to lose.

3. **Audit product bundling practices in Germany** — the combination of high churn rates and product overloading suggests a structural issue that marketing campaigns alone cannot fix.

4. **Implement an engagement scoring system** in the core banking platform to automatically flag customers who go inactive, triggering intervention workflows before the 6-month disengagement threshold.

5. **Retrain the model quarterly** as customer behavior evolves. The pipeline is fully automated and designed for periodic refresh.

---

*Churn Intelligence System · Created by Anoop Puri*
*For internal risk management and retention strategy use only*
