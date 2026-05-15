# Deployment Guide
## Churn Intelligence System — GitHub + Streamlit Cloud

---

## Step 1 — Upload to GitHub

### Option A: GitHub Desktop (easiest)
1. Download and install [GitHub Desktop](https://desktop.github.com)
2. Click **File → Add local repository**
3. Browse to the extracted `churn_intelligence/` folder
4. Click **Publish repository** → give it a name → **Publish**

### Option B: GitHub Web Upload
1. Go to [github.com/new](https://github.com/new)
2. Name your repo (e.g. `churn-intelligence`) → **Create repository**
3. On the next page click **uploading an existing file**
4. Drag the entire unzipped `churn_intelligence/` folder contents in
5. Commit message: `Initial commit — Churn Intelligence System v2`
6. Click **Commit changes**

### Option C: Git Command Line
```bash
# 1. Extract the zip, then cd into it
cd churn_intelligence

# 2. Initialize and push
git init
git add .
git commit -m "Initial commit — Churn Intelligence System v2"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/churn-intelligence.git
git push -u origin main
```

> **Important about .pkl model files**: The trained model files in `models/` are ~15MB total. GitHub allows files up to 100MB, so they will upload fine without Git LFS. If you later get a warning, see the Git LFS section below.

---

## Step 2 — Deploy on Streamlit Cloud (Free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** → Sign in with GitHub
2. Click **New app**
3. Fill in:
   - **Repository**: `YOUR_USERNAME/churn-intelligence`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**
5. Wait ~3 minutes for dependencies to install
6. Your app is live at: `https://YOUR_USERNAME-churn-intelligence-app-XXXXX.streamlit.app`

### If deployment fails — common fixes:

**Error: Module not found**
```
# Make sure requirements.txt is in the root of your repo (it already is)
```

**Error: File not found (models/*.pkl)**
```
# The models/ folder must be committed to GitHub — not in .gitignore
# Open .gitignore and confirm *.pkl is NOT listed (it isn't)
```

**Error: Port already in use (local only)**
```bash
streamlit run app.py --server.port 8502
```

---

## Step 3 — Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Retrain all models (optional — pre-trained models already included)
python src/train_pipeline.py

# Rebuild Excel workbook
python excel/build_excel_model.py
```

---

## File Structure After Extraction

```
churn_intelligence/           ← Root of your GitHub repo
│
├── app.py                    ← Streamlit Cloud entry point (set this as Main file)
├── requirements.txt          ← All Python dependencies
├── packages.txt              ← System packages (empty — not needed)
├── README.md                 ← Full documentation with badges
├── LICENSE                   ← MIT License
├── .gitignore
│
├── .streamlit/
│   └── config.toml           ← Theme + server config for Streamlit Cloud
│
├── .github/
│   └── workflows/
│       └── ci.yml            ← GitHub Actions — auto-tests on every push
│
├── app/
│   └── streamlit_app.py      ← Full 7-page dashboard (1,612 lines)
│
├── src/
│   ├── preprocessing.py      ← Feature engineering + SMOTE
│   ├── models.py             ← 5 ML models + calibration
│   ├── evaluation.py         ← Metrics + threshold analysis
│   ├── eda.py                ← EDA statistics module
│   ├── explainability.py     ← SHAP computation
│   └── train_pipeline.py     ← One-command full training run
│
├── models/                   ← Pre-trained model artifacts
│   ├── gradient_boosting.pkl ← Best model (AUC 0.8671)
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── feature_names.json
│
├── outputs/                  ← Pre-computed results
│   ├── scored_customers.csv  ← All 10,000 customers with risk scores
│   ├── model_comparison.csv  ← All 5 model metrics
│   ├── feature_importance.csv
│   ├── shap_importance.csv
│   ├── model_results.json
│   └── summary_stats.json
│
├── data/
│   └── European_Bank.csv     ← Source data (10,000 records)
│
├── excel/
│   ├── Churn_Intelligence_Model.xlsx  ← 7-sheet institutional model
│   └── build_excel_model.py           ← Rebuild script
│
└── docs/
    ├── research_paper.md     ← Full academic paper (396 lines)
    └── executive_summary.md  ← Board/government summary
```

---

## Git LFS (Only if GitHub rejects .pkl files)

If GitHub warns about large files:

```bash
# Install Git LFS
git lfs install

# Track pkl files
git lfs track "*.pkl"
git add .gitattributes
git add models/*.pkl
git commit -m "Add models with Git LFS"
git push
```

---

## Dashboard Pages Reference

| Page | Sidebar Label |
|---|---|
| Executive summary + KPIs | 🏠 Executive Dashboard |
| Box plots, heatmaps, distributions | 🔬 EDA & Analytics |
| Individual customer scorer | 🎯 Risk Calculator |
| ROC, radar, confusion matrices | 📈 Model Performance |
| Tree + SHAP importance, correlations | 🔍 Feature Intelligence |
| Retention scenario simulator | 🔬 What-If Simulator |
| Filterable 10,000-row register | 📋 Customer Register |

---

*Churn Intelligence System v2 — Created by Anoop Puri*
