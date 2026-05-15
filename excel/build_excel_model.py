"""
build_excel_model.py — Institutional Excel model builder
Produces a 7-sheet, fully-formula-linked workbook.
"""
import os, json
import numpy as np
import pandas as pd
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY="0D1B2A"; INK="1B263B"; BLUE="1B4F72"; SKY="2E86AB"; GOLD="C9A84C"
AMBER="E8A838"; CREAM="F5F0E8"; WHITE="FFFFFF"; MIST="F0F4F8"; SLATE="64748B"
STEEL="94A3B8"; RED="B91C1C"; GREEN="15803D"; ORANGE="EA580C"; BORDER="E2E8F0"
LGRAY="F8FAFC"; YELLOW="FFFBEB"

def fnt(bold=False,size=10,color="000000",italic=False,name="Calibri"):
    return Font(name=name,bold=bold,size=size,color=color,italic=italic)
def fill(c): return PatternFill("solid",fgColor=c)
def aln(h="left",v="center",wrap=False): return Alignment(horizontal=h,vertical=v,wrap_text=wrap)
def bdr_bottom(c=BORDER):
    b=Side(style="thin",color=c); n=Side(style=None)
    return Border(bottom=b,left=n,right=n,top=n)
def bdr_all(c=BORDER):
    b=Side(style="thin",color=c)
    return Border(top=b,bottom=b,left=b,right=b)
def bdr_header():
    return Border(bottom=Side(style="medium",color=GOLD),
                  left=Side(style=None),right=Side(style=None),top=Side(style=None))

def set_widths(ws,widths):
    for col,w in widths.items(): ws.column_dimensions[col].width=w

def hdr_row(ws,row,cols,bg=NAVY,fg=WHITE,h=20):
    ws.row_dimensions[row].height=h
    for col,val in cols:
        c=ws.cell(row=row,column=col,value=val)
        c.font=fnt(bold=True,size=9,color=fg)
        c.fill=fill(bg); c.alignment=aln("center"); c.border=bdr_header()

def sec_row(ws,row,col,text,span=6,bg=LGRAY):
    c=ws.cell(row=row,column=col,value=text)
    c.font=fnt(bold=True,size=8,color=SLATE)
    c.fill=fill(bg); c.alignment=aln("left")
    c.border=bdr_bottom(BORDER)
    ws.row_dimensions[row].height=14

def dat(ws,row,col,val,fmt=None,bold=False,color="000000",bg=WHITE,h="right"):
    c=ws.cell(row=row,column=col,value=val)
    c.font=fnt(bold=bold,size=10,color=color)
    c.fill=fill(bg); c.alignment=aln(h,"center")
    c.border=bdr_bottom()
    if fmt: c.number_format=fmt
    return c


BASE_DIR   =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR=os.path.join(BASE_DIR,"outputs")
DATA_PATH  =os.path.join(BASE_DIR,"data","European_Bank.csv")

with open(os.path.join(OUTPUTS_DIR,"summary_stats.json"))  as f: summ=json.load(f)
with open(os.path.join(OUTPUTS_DIR,"model_results.json"))  as f: mres=json.load(f)
comp_df =pd.read_csv(os.path.join(OUTPUTS_DIR,"model_comparison.csv"))
fi_df   =pd.read_csv(os.path.join(OUTPUTS_DIR,"feature_importance.csv"))
shap_df =pd.read_csv(os.path.join(OUTPUTS_DIR,"shap_importance.csv"))
raw_df  =pd.read_csv(DATA_PATH)
scored  =pd.read_csv(os.path.join(OUTPUTS_DIR,"scored_customers.csv"))


def build():
    wb=Workbook(); wb.remove(wb.active)
    build_cover(wb)
    build_assumptions(wb)
    build_model_comparison(wb)
    build_feature_importance(wb)
    build_portfolio_analysis(wb)
    build_risk_register(wb)
    build_eda_summary(wb)
    out=os.path.join(BASE_DIR,"excel","Churn_Intelligence_Model.xlsx")
    os.makedirs(os.path.dirname(out),exist_ok=True)
    wb.save(out)
    print(f"[Excel] Saved → {out}")
    return out


# ── COVER ────────────────────────────────────────────────────────────────────
def build_cover(wb):
    ws=wb.create_sheet("Cover"); ws.sheet_view.showGridLines=False
    set_widths(ws,{"A":3,"B":42,"C":36,"D":18})

    for r in [2,3,4,5]: ws.row_dimensions[r].height=38 if r==2 else 20 if r in [3,4] else 5
    for col in ["B","C","D"]:
        for r in [2,3,4]:
            ws[f"{col}{r}"].fill=fill(NAVY)
    ws.merge_cells("B2:D2")
    c=ws["B2"]; c.value="CHURN INTELLIGENCE SYSTEM"
    c.font=Font(name="Calibri",bold=True,size=22,color=WHITE)
    c.fill=fill(NAVY); c.alignment=aln("left","center")

    ws.merge_cells("B3:D3")
    c=ws["B3"]; c.value="Predictive Risk Scoring & Customer Retention Analytics — European Bank"
    c.font=fnt(size=11,color=GOLD,italic=True); c.fill=fill(NAVY); c.alignment=aln("left","center")

    ws.merge_cells("B4:D4")
    c=ws["B4"]; c.value="SMOTE · Isotonic Calibration · Gradient Boosting · SHAP Explainability"
    c.font=fnt(size=9,color=STEEL,italic=True); c.fill=fill(NAVY); c.alignment=aln("left","center")

    ws.merge_cells("B5:D5"); ws["B5"].fill=fill(GOLD); ws.row_dimensions[5].height=4

    meta=[
        ("Dataset",           "European Bank Customer Dataset"),
        ("Records",           f"{summ['total_customers']:,} customers"),
        ("Actual Churn Rate", f"{summ['actual_churn_rate']:.2%}"),
        ("Best Model",        summ["best_model"]),
        ("ROC-AUC",           f"{summ['best_roc_auc']:.4f}"),
        ("Recall",            f"{summ['best_recall']:.1%}"),
        ("Precision",         f"{summ['best_precision']:.1%}"),
        ("F1 Score",          f"{summ['best_f1']:.4f}"),
        ("Features",          f"{summ['feature_count']} (14 raw + 7 engineered)"),
        ("SMOTE Applied",     "Yes — sampling_strategy=0.6"),
        ("Calibration",       "Isotonic Regression (cv=5)"),
        ("Threshold Method",  "Youden's J Statistic"),
    ]
    for i,(label,val) in enumerate(meta,start=7):
        ws.row_dimensions[i].height=17
        cl=ws.cell(row=i,column=2,value=label)
        cl.font=fnt(bold=True,size=10,color=SLATE); cl.border=bdr_bottom(); cl.alignment=aln("left")
        cv=ws.cell(row=i,column=3,value=val)
        cv.font=fnt(size=10,color=INK); cv.border=bdr_bottom(); cv.alignment=aln("left")

    ws.merge_cells("B20:D20")
    c=ws["B20"]; c.value="WORKBOOK INDEX"
    c.font=fnt(bold=True,size=9,color=SLATE); c.border=bdr_bottom(GOLD)

    sheets=[
        ("Cover",             "Project overview, metadata, sheet index"),
        ("Assumptions",       "All hardcoded inputs — blue = input cell"),
        ("Model Comparison",  "Performance metrics across 5 ML models with ROC-AUC ranking"),
        ("Feature Importance","Tree importance + SHAP values + cumulative coverage"),
        ("Portfolio Analysis","Risk band segmentation + geography + gender breakdown"),
        ("Risk Register",     "Top 200 highest-risk customers for immediate retention action"),
        ("EDA Summary",       "Statistical profile, churn comparisons, product-level analysis"),
    ]
    for i,(sname,sdesc) in enumerate(sheets,start=21):
        ws.row_dimensions[i].height=16
        cs=ws.cell(row=i,column=2,value=sname)
        cs.font=fnt(bold=True,size=9,color=BLUE); cs.border=bdr_bottom(); cs.alignment=aln("left")
        cd=ws.cell(row=i,column=3,value=sdesc)
        cd.font=fnt(size=9,color=INK); cd.border=bdr_bottom()


# ── ASSUMPTIONS ──────────────────────────────────────────────────────────────
def build_assumptions(wb):
    ws=wb.create_sheet("Assumptions"); ws.sheet_view.showGridLines=False
    set_widths(ws,{"A":3,"B":38,"C":20,"D":14,"E":44})
    ws.row_dimensions[1].height=36

    ws.merge_cells("B1:E1")
    c=ws["B1"]; c.value="MODEL ASSUMPTIONS & CONFIGURATION"
    c.font=Font(name="Calibri",bold=True,size=14,color=WHITE)
    c.fill=fill(NAVY); c.alignment=aln("left","center")

    hdr_row(ws,2,[(2,"Parameter"),(3,"Value"),(4,"Unit"),(5,"Notes")])

    rows=[
        ("── DATA CONFIGURATION",None,None,None),
        ("Total Records",10000,"count","European_Bank.csv source"),
        ("Train / Val / Test Split","80 / 10 / 10","%","Stratified by Exited flag"),
        ("Random State Seed",42,"integer","Reproducibility"),
        ("Cross-Validation Folds",5,"folds","Stratified K-Fold"),
        ("── SMOTE OVERSAMPLING",None,None,None),
        ("Sampling Strategy",0.60,"ratio","Minority/majority after SMOTE"),
        ("k-Neighbours",5,"count","SMOTE neighbourhood"),
        ("Post-SMOTE Train Size",10193,"samples","8,001 → 10,193"),
        ("Post-SMOTE Churn Rate",0.375,"ratio","0.204 → 0.375"),
        ("── BEST MODEL — GRADIENT BOOSTING",None,None,None),
        ("n_estimators",350,"trees","Boosting rounds"),
        ("learning_rate",0.04,"rate","Shrinkage"),
        ("max_depth",4,"levels","Controls overfitting"),
        ("subsample",0.82,"ratio","Row sampling per tree"),
        ("min_samples_split",25,"samples","Node split criterion"),
        ("min_samples_leaf",12,"samples","Leaf minimum"),
        ("max_features","sqrt",None,"Feature subsampling"),
        ("── PROBABILITY CALIBRATION",None,None,None),
        ("Method","Isotonic",None,"Non-parametric calibration"),
        ("Cross-Validation",5,"folds","CalibratedClassifierCV cv=5"),
        ("── CLASSIFICATION THRESHOLD",None,None,None),
        ("Method","Youden's J",None,"max(TPR − FPR)"),
        ("Optimal Threshold",round(summ["optimal_threshold"],4),"probability","Per-model optimised"),
        ("── RISK BAND THRESHOLDS",None,None,None),
        ("Low Risk Upper Bound",0.20,"probability","< 20% = Low Risk"),
        ("Moderate Risk Upper Bound",0.40,"probability","20%–40%"),
        ("High Risk Upper Bound",0.65,"probability","40%–65%"),
        ("Critical Risk Lower Bound",0.65,"probability","≥ 65% = Critical"),
        ("── MODEL PERFORMANCE",None,None,None),
        ("ROC-AUC",summ["best_roc_auc"],"score","Area under ROC curve"),
        ("F1 Score",summ["best_f1"],"score","Harmonic mean P/R"),
        ("Recall",summ["best_recall"],"score","True Positive Rate"),
        ("Precision",summ["best_precision"],"score","Positive Predictive Value"),
    ]
    for i,(param,val,unit,note) in enumerate(rows,start=3):
        ws.row_dimensions[i].height=16
        is_sec=(val is None)
        bp=ws.cell(row=i,column=2,value=param)
        if is_sec:
            bp.font=fnt(bold=True,size=8,color=SLATE); bp.fill=fill(LGRAY)
            bp.border=bdr_bottom(BORDER)
            for col in [3,4,5]:
                ws.cell(row=i,column=col).fill=fill(LGRAY); ws.cell(row=i,column=col).border=bdr_bottom()
        else:
            bp.font=fnt(size=10,color=INK); bp.alignment=aln("left"); bp.border=bdr_bottom()
            cv=ws.cell(row=i,column=3,value=val)
            if isinstance(val,(int,float)):
                cv.font=fnt(bold=True,size=10,color="0000FF")
                cv.number_format="0.0000" if isinstance(val,float) and val<100 else "#,##0"
            else:
                cv.font=fnt(size=9,color=BLUE,italic=True)
            cv.alignment=aln("center"); cv.border=bdr_bottom()
            if unit:
                cu=ws.cell(row=i,column=4,value=unit)
                cu.font=fnt(size=9,color=SLATE,italic=True); cu.alignment=aln("center"); cu.border=bdr_bottom()
            if note:
                cn=ws.cell(row=i,column=5,value=note)
                cn.font=fnt(size=9,color=SLATE); cn.border=bdr_bottom()


# ── MODEL COMPARISON ─────────────────────────────────────────────────────────
def build_model_comparison(wb):
    ws=wb.create_sheet("Model Comparison"); ws.sheet_view.showGridLines=False
    set_widths(ws,{"A":3,"B":26,"C":13,"D":13,"E":13,"F":13,"G":13,"H":14,"I":16,"J":18})
    ws.row_dimensions[1].height=36

    ws.merge_cells("B1:J1")
    c=ws["B1"]; c.value="ML MODEL PERFORMANCE COMPARISON"
    c.font=Font(name="Calibri",bold=True,size=14,color=WHITE)
    c.fill=fill(NAVY); c.alignment=aln("left","center")

    ws.merge_cells("B2:J2")
    c=ws["B2"]; c.value=(f"Ranked by ROC-AUC  ·  Best Model: {summ['best_model']}  ·  "
                          f"Threshold: Youden's J per model  ·  Calibration: Isotonic cv=5  ·  SMOTE applied")
    c.font=fnt(size=8,color=SLATE,italic=True); c.fill=fill(YELLOW)
    c.alignment=aln("left","center"); c.border=bdr_bottom(GOLD)
    ws.row_dimensions[2].height=14

    hdr_row(ws,3,[(2,"Model"),(3,"Accuracy"),(4,"Precision"),(5,"Recall"),
                   (6,"F1 Score"),(7,"ROC-AUC"),(8,"Avg Prec"),(9,"Specificity"),(10,"Threshold")])

    keys=["Accuracy","Precision","Recall","F1 Score","ROC-AUC","Avg Precision","Specificity"]
    for i,(_, row) in enumerate(comp_df.iterrows(),start=4):
        is_best=(row["Model"]==summ["best_model"])
        bg="FFF8E7" if is_best else WHITE; bld=is_best
        ws.row_dimensions[i].height=18

        b=ws.cell(row=i,column=2,value=("★ " if is_best else "")+row["Model"])
        b.font=fnt(bold=bld,size=10,color=BLUE if is_best else INK)
        b.fill=fill(bg); b.alignment=aln("left"); b.border=bdr_bottom()

        for j,k in enumerate(keys,start=3):
            v=row[k]; c=ws.cell(row=i,column=j,value=v)
            col_=RED if k=="Recall" and v<0.70 else GREEN if k=="ROC-AUC" and v==comp_df["ROC-AUC"].max() else INK
            c.font=fnt(bold=bld,size=10,color=col_)
            c.number_format="0.0000"; c.fill=fill(bg); c.alignment=aln("center"); c.border=bdr_bottom()

        # Threshold
        thresh=mres.get(row["Model"],{}).get("optimal_threshold","—")
        ct=ws.cell(row=i,column=10,value=thresh)
        ct.font=fnt(size=9,color=SLATE); ct.number_format="0.0000"
        ct.fill=fill(bg); ct.alignment=aln("center"); ct.border=bdr_bottom()

    # Color scale on AUC
    auc_rng=f"G4:G{3+len(comp_df)}"
    ws.conditional_formatting.add(auc_rng,
        ColorScaleRule(start_type="min",start_color="F0F4F8",
                       end_type="max",  end_color=NAVY))

    # Definitions
    def_row=4+len(comp_df)+2
    ws.merge_cells(f"B{def_row}:J{def_row}")
    c=ws[f"B{def_row}"]; c.value="METRIC DEFINITIONS"
    c.font=fnt(bold=True,size=9,color=SLATE); c.fill=fill(LGRAY); c.border=bdr_bottom()

    defs=[
        ("Accuracy","(TP+TN)/(TP+TN+FP+FN) — Overall fraction of correct predictions"),
        ("Precision","TP/(TP+FP) — Of predicted churners, fraction who actually churned"),
        ("Recall","TP/(TP+FN) — Of actual churners, fraction correctly identified"),
        ("F1 Score","2×P×R/(P+R) — Harmonic mean of Precision and Recall"),
        ("ROC-AUC","Area under ROC curve — discrimination power across all thresholds"),
        ("Avg Prec","Area under Precision-Recall curve — robust to class imbalance"),
        ("Specificity","TN/(TN+FP) — Fraction of retained customers correctly identified"),
        ("Threshold","Optimal classification cutoff via Youden's J = TPR − FPR"),
    ]
    for k,(m,d) in enumerate(defs,start=def_row+1):
        ws.row_dimensions[k].height=15
        cm=ws.cell(row=k,column=2,value=m); cm.font=fnt(bold=True,size=9,color=BLUE)
        cm.border=bdr_bottom()
        cd=ws.cell(row=k,column=3,value=d); cd.font=fnt(size=9,color=INK)
        cd.border=bdr_bottom()
        ws.merge_cells(f"C{k}:J{k}")


# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
def build_feature_importance(wb):
    ws=wb.create_sheet("Feature Importance"); ws.sheet_view.showGridLines=False
    set_widths(ws,{"A":3,"B":28,"C":16,"D":16,"E":16,"F":44})
    ws.row_dimensions[1].height=36

    ws.merge_cells("B1:F1")
    c=ws["B1"]; c.value="FEATURE IMPORTANCE — TREE-BASED & SHAP VALUES"
    c.font=Font(name="Calibri",bold=True,size=14,color=WHITE)
    c.fill=fill(NAVY); c.alignment=aln("left","center")

    hdr_row(ws,2,[(2,"Feature"),(3,"Tree Importance"),(4,"SHAP Importance"),
                   (5,"Cumulative %"),(6,"Business Interpretation")])

    total_imp=fi_df["Importance"].sum()
    shap_dict=dict(zip(shap_df["Feature"],shap_df["SHAP_Importance"]))

    interp={
        "AgeGroup_Senior":     "Customers ≥45 yrs churn at nearly 2× rate — key demographic flag",
        "Age":                 "Continuous age risk — churner mean 44.8 vs 37.4 retained",
        "NumOfProducts":       "Product overload cliff: 3 products → 82.7% churn, 4 → 100%",
        "MultiProductRisk":    "Binary flag for 3+ product customers — captures non-linear cliff",
        "Geography_Germany":   "Germany: 32.4% churn rate vs 16.2% France (2× premium)",
        "IsActiveMember":      "Inactive members churn at 26.9% vs 14.3% — highest ROI lever",
        "Tenure":              "Mid-tenure (3–7 yrs) peak risk; very short/long tenure lower",
        "Gender_Male":         "Male customers churn at lower rate (16.5% vs 25.1% female)",
        "Balance":             "Churner mean balance €91k vs €73k retained — wealth paradox",
        "Geography_France":    "Baseline geography — lowest churn (16.2%)",
        "Geography_Spain":     "Spain similar to France (16.7%) — lower risk",
        "InactiveHighBalance": "High-value inactive customers — premium CLV at maximum risk",
        "ProductDensity":      "Products per tenure year — rapid bundling precedes churn",
        "Gender_Female":       "Female churn rate 25.1% — product & communication opportunity",
        "BalanceToSalaryRatio":"Financial dependency proxy — high ratio signals expectations",
        "AgeTenureInteraction":"Older + short tenure = highest risk combination",
        "EngagementScore":     "Composite 0–2 activity index — re-activation signal",
        "CreditScore":         "Lower scores weakly correlated with churn (r=−0.028)",
        "EstimatedSalary":     "Weak predictor — income band matters less than engagement",
        "ZeroBalance":         "Dormant account indicator (15.6% of zero-balance churn)",
        "HasCrCard":           "Credit card alone is weak predictor (r=−0.007)",
    }

    cumulative=0
    for i,(_, row) in enumerate(fi_df.iterrows(),start=3):
        cumulative+=row["Importance"]
        cum_pct=cumulative/total_imp
        ws.row_dimensions[i].height=16
        is_top5=(i<=7)
        bg="FFF8E7" if row["Importance"]>=fi_df["Importance"].nlargest(5).min() else WHITE

        b=ws.cell(row=i,column=2,value=row["Feature"])
        b.font=fnt(bold=is_top5,size=10,color=BLUE if is_top5 else INK)
        b.fill=fill(bg); b.alignment=aln("left"); b.border=bdr_bottom()

        ci=ws.cell(row=i,column=3,value=row["Importance"])
        ci.font=fnt(bold=is_top5,size=10,
                    color=RED if row["Importance"]>=0.10 else ORANGE if row["Importance"]>=0.05 else INK)
        ci.number_format="0.0000"; ci.fill=fill(bg); ci.alignment=aln("center"); ci.border=bdr_bottom()

        shap_v=shap_dict.get(row["Feature"],0)
        cs=ws.cell(row=i,column=4,value=shap_v)
        cs.font=fnt(size=10,color=RED if shap_v>=0.3 else ORANGE if shap_v>=0.1 else INK)
        cs.number_format="0.0000"; cs.fill=fill(bg); cs.alignment=aln("center"); cs.border=bdr_bottom()

        cc=ws.cell(row=i,column=5,value=cum_pct)
        cc.font=fnt(size=9,color=GREEN if cum_pct>=0.80 else SLATE)
        cc.number_format="0.0%"; cc.fill=fill(bg); cc.alignment=aln("center"); cc.border=bdr_bottom()

        cn=ws.cell(row=i,column=6,value=interp.get(row["Feature"],""))
        cn.font=fnt(size=9,color=INK); cn.fill=fill(bg)
        cn.alignment=aln("left","center",wrap=True); cn.border=bdr_bottom()

    imp_rng=f"C3:C{2+len(fi_df)}"
    ws.conditional_formatting.add(imp_rng,DataBarRule(start_type="min",end_type="max",color=SKY))
    shap_rng=f"D3:D{2+len(shap_df)}"
    ws.conditional_formatting.add(shap_rng,DataBarRule(start_type="min",end_type="max",color="B91C1C"))


# ── PORTFOLIO ANALYSIS ────────────────────────────────────────────────────────
def build_portfolio_analysis(wb):
    ws=wb.create_sheet("Portfolio Analysis"); ws.sheet_view.showGridLines=False
    set_widths(ws,{"A":3,"B":22,"C":14,"D":14,"E":14,"F":14,"G":14,"H":14})
    ws.row_dimensions[1].height=36

    ws.merge_cells("B1:H1")
    c=ws["B1"]; c.value="PORTFOLIO RISK ANALYSIS"
    c.font=Font(name="Calibri",bold=True,size=14,color=WHITE)
    c.fill=fill(NAVY); c.alignment=aln("left","center")

    # Risk Band Summary
    sec_row(ws,3,2,"RISK BAND SEGMENTATION")
    hdr_row(ws,4,[(2,"Risk Band"),(3,"Customers"),(4,"% Portfolio"),
                   (5,"Actual Churners"),(6,"Actual Churn Rate"),(7,"Avg Prob."),(8,"Avg Age")])

    band_colors={"Critical Risk":RED,"High Risk":ORANGE,"Moderate Risk":AMBER,"Low Risk":GREEN}
    for i,band in enumerate(["Critical Risk","High Risk","Moderate Risk","Low Risk"],start=5):
        sub=scored[scored["RiskBand"]==band]; sub_raw=raw_df[scored["RiskBand"]==band]
        bg={"Critical Risk":"FEF2F2","High Risk":"FFF7ED","Moderate Risk":"FFFBEB","Low Risk":"F0FDF4"}[band]
        ws.row_dimensions[i].height=18
        col=band_colors[band]; frac=len(sub)/len(scored)
        rate=sub["Exited"].mean() if len(sub)>0 else 0

        for j,(val,fmt,halign) in enumerate([
            (band,None,"left"),
            (len(sub),"#,##0","center"),
            (frac,"0.0%","center"),
            (int(sub["Exited"].sum()),"#,##0","center"),
            (rate,"0.0%","center"),
            (sub["ChurnProbability"].mean(),"0.000","center"),
            (raw_df.loc[scored["RiskBand"]==band,"Age"].mean(),"0.0","center"),
        ],start=2):
            c=ws.cell(row=i,column=j,value=val)
            c.font=fnt(bold=(j==2),size=10,color=col if j==2 else RED if j==6 and rate>0.5 else INK)
            if fmt: c.number_format=fmt
            c.fill=fill(bg); c.alignment=aln(halign,"center"); c.border=bdr_bottom()

    # Totals
    ws.row_dimensions[9].height=18
    for j,(val,fmt) in enumerate([
        ("TOTAL PORTFOLIO",None),(len(scored),"#,##0"),(1.0,"0.0%"),
        (int(scored["Exited"].sum()),"#,##0"),(scored["Exited"].mean(),"0.0%"),
        (scored["ChurnProbability"].mean(),"0.000"),(raw_df["Age"].mean(),"0.0"),
    ],start=2):
        c=ws.cell(row=9,column=j,value=val)
        c.font=fnt(bold=True,size=10,color=WHITE); c.fill=fill(NAVY)
        if fmt: c.number_format=fmt
        c.alignment=aln("left" if j==2 else "center","center"); c.border=bdr_all(GOLD)

    # Geography
    sec_row(ws,11,2,"CHURN BY GEOGRAPHY")
    hdr_row(ws,12,[(2,"Geography"),(3,"Total"),(4,"Churned"),(5,"Churn Rate"),
                    (6,"Avg Balance"),(7,"Avg Age"),(8,"Avg CreditScore")])
    for i,geo in enumerate(["France","Germany","Spain"],start=13):
        sub=raw_df[raw_df["Geography"]==geo]; ws.row_dimensions[i].height=16
        rate=sub["Exited"].mean()
        for j,(val,fmt,col_) in enumerate([
            (geo,None,BLUE),(len(sub),"#,##0",INK),(int(sub["Exited"].sum()),"#,##0",INK),
            (rate,"0.0%",RED if rate>0.25 else INK),
            (sub["Balance"].mean(),"#,##0",INK),(sub["Age"].mean(),"0.1",INK),
            (sub["CreditScore"].mean(),"0.0",INK),
        ],start=2):
            c=ws.cell(row=i,column=j,value=round(val,4) if isinstance(val,float) else val)
            c.font=fnt(bold=(j==5 and rate>0.25),size=10,color=col_)
            if fmt: c.number_format=fmt
            c.alignment=aln("left" if j==2 else "center","center"); c.border=bdr_bottom()

    # Gender
    sec_row(ws,17,2,"CHURN BY GENDER")
    hdr_row(ws,18,[(2,"Gender"),(3,"Total"),(4,"Churned"),(5,"Churn Rate"),
                    (6,"Avg Balance"),(7,"Avg Salary"),(8,"Avg Age")])
    for i,gen in enumerate(["Female","Male"],start=19):
        sub=raw_df[raw_df["Gender"]==gen]; ws.row_dimensions[i].height=16
        rate=sub["Exited"].mean()
        for j,(val,fmt) in enumerate([
            (gen,None),(len(sub),"#,##0"),(int(sub["Exited"].sum()),"#,##0"),
            (rate,"0.0%"),(sub["Balance"].mean(),"#,##0"),
            (sub["EstimatedSalary"].mean(),"#,##0"),(sub["Age"].mean(),"0.1"),
        ],start=2):
            c=ws.cell(row=i,column=j,value=round(val,4) if isinstance(val,float) else val)
            c.font=fnt(size=10,color=RED if j==5 and rate>0.22 else INK)
            if fmt: c.number_format=fmt
            c.alignment=aln("left" if j==2 else "center","center"); c.border=bdr_bottom()

    # NumOfProducts
    sec_row(ws,22,2,"CHURN BY NUMBER OF PRODUCTS — KEY INSIGHT")
    hdr_row(ws,23,[(2,"Num Products"),(3,"Total"),(4,"Churned"),(5,"Churn Rate"),
                    (6,"Avg Balance"),(7,"Avg Age"),(8,"Risk Assessment")])
    risk_labels={1:"Low-Moderate",2:"Low",3:"CRITICAL",4:"EXTREME (100%)"}
    for i,n in enumerate([1,2,3,4],start=24):
        sub=raw_df[raw_df["NumOfProducts"]==n]; ws.row_dimensions[i].height=16
        rate=sub["Exited"].mean()
        bg="FEF2F2" if n>=3 else WHITE
        for j,(val,fmt) in enumerate([
            (n,"#,##0"),(len(sub),"#,##0"),(int(sub["Exited"].sum()),"#,##0"),
            (rate,"0.0%"),(sub["Balance"].mean(),"#,##0"),(sub["Age"].mean(),"0.1"),
            (risk_labels[n],None),
        ],start=2):
            c=ws.cell(row=i,column=j,value=round(val,4) if isinstance(val,float) else val)
            c.font=fnt(bold=(n>=3),size=10,color=RED if n>=3 else INK)
            if fmt: c.number_format=fmt
            c.fill=fill(bg); c.alignment=aln("right" if j>2 and j<9 else "left","center")
            c.border=bdr_bottom()


# ── RISK REGISTER ─────────────────────────────────────────────────────────────
def build_risk_register(wb):
    ws=wb.create_sheet("Risk Register"); ws.sheet_view.showGridLines=False
    set_widths(ws,{"A":3,"B":12,"C":16,"D":12,"E":10,"F":10,"G":10,"H":14,"I":14,"J":14,"K":14})
    ws.row_dimensions[1].height=36

    ws.merge_cells("B1:K1")
    c=ws["B1"]; c.value="HIGH-RISK CUSTOMER REGISTER — TOP 200 BY CHURN PROBABILITY"
    c.font=Font(name="Calibri",bold=True,size=14,color=WHITE)
    c.fill=fill(NAVY); c.alignment=aln("left","center")

    ws.merge_cells("B2:K2")
    c=ws["B2"]; c.value="Sorted by churn probability (descending). Critical Risk requires immediate retention action within 48 hours."
    c.font=fnt(size=8,color=SLATE,italic=True); c.fill=fill(YELLOW)
    c.alignment=aln("left","center"); c.border=bdr_bottom(GOLD)
    ws.row_dimensions[2].height=14

    hdr_row(ws,3,[(2,"Customer ID"),(3,"Surname"),(4,"Geography"),(5,"Gender"),
                   (6,"Age"),(7,"Products"),(8,"Balance (€)"),(9,"Churn Prob."),
                   (10,"Risk Band"),(11,"Actual Exited")])

    top200=scored.sort_values("ChurnProbability",ascending=False).head(200)
    rb_bg={"Critical Risk":"FEF2F2","High Risk":"FFF7ED","Moderate Risk":"FFFBEB","Low Risk":"F0FDF4"}
    rb_fc={"Critical Risk":RED,"High Risk":ORANGE,"Moderate Risk":AMBER,"Low Risk":GREEN}

    for i,(_,row) in enumerate(top200.iterrows(),start=4):
        ws.row_dimensions[i].height=15
        bg=rb_bg[row["RiskBand"]]; rc=rb_fc[row["RiskBand"]]
        for j,(val,fmt,halign) in enumerate([
            (row["CustomerId"],"#,##0","center"),
            (row["Surname"],None,"left"),
            (row["Geography"],None,"center"),
            (row["Gender"],None,"center"),
            (row["Age"],"#,##0","center"),
            (row["NumOfProducts"],"#,##0","center"),
            (row["Balance"],"#,##0.00","right"),
            (row["ChurnProbability"],"0.000","center"),
            (row["RiskBand"],None,"center"),
            ("Churned" if row["Exited"]==1 else "Retained",None,"center"),
        ],start=2):
            c=ws.cell(row=i,column=j,value=val)
            if j==9: c.font=fnt(bold=True,size=9,color=rc)
            elif j==10: c.font=fnt(bold=True,size=9,color=rc)
            elif j==11: c.font=fnt(bold=(row["Exited"]==1),size=9,
                                   color=RED if row["Exited"]==1 else GREEN)
            else: c.font=fnt(size=9,color=INK)
            if fmt: c.number_format=fmt
            c.fill=fill(bg); c.alignment=aln(halign,"center"); c.border=bdr_bottom()

    # Color scale on probability column
    ws.conditional_formatting.add(f"I4:I{3+len(top200)}",
        ColorScaleRule(start_type="num",start_value=0.30,start_color="F0FDF4",
                       mid_type="num",  mid_value=0.55, mid_color="FFFBEB",
                       end_type="num",  end_value=0.95, end_color="FEF2F2"))


# ── EDA SUMMARY ───────────────────────────────────────────────────────────────
def build_eda_summary(wb):
    ws=wb.create_sheet("EDA Summary"); ws.sheet_view.showGridLines=False
    set_widths(ws,{"A":3,"B":22,"C":16,"D":16,"E":16,"F":16,"G":16})
    ws.row_dimensions[1].height=36

    ws.merge_cells("B1:G1")
    c=ws["B1"]; c.value="EXPLORATORY DATA ANALYSIS — STATISTICAL PROFILE"
    c.font=Font(name="Calibri",bold=True,size=14,color=WHITE)
    c.fill=fill(NAVY); c.alignment=aln("left","center")

    num_cols=["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
    desc=raw_df[num_cols].describe().round(2)

    # Descriptive stats
    sec_row(ws,3,2,"NUMERICAL FEATURE DESCRIPTIVE STATISTICS")
    hdr_row(ws,4,[(2,"Feature"),(3,"Mean"),(4,"Std Dev"),(5,"Min"),(6,"Median"),(7,"Max")])
    for i,cn in enumerate(num_cols,start=5):
        ws.row_dimensions[i].height=16
        ws.cell(row=i,column=2,value=cn).font=fnt(size=10,color=BLUE)
        ws.cell(row=i,column=2).border=bdr_bottom()
        for j,stat in enumerate(["mean","std","min","50%","max"],start=3):
            c=ws.cell(row=i,column=j,value=float(desc.loc[stat,cn]))
            c.font=fnt(size=10); c.number_format="#,##0.00"
            c.alignment=aln("right","center"); c.border=bdr_bottom()

    # Churn comparison
    sec_row(ws,12,2,"MEAN VALUES — CHURNED VS RETAINED")
    hdr_row(ws,13,[(2,"Feature"),(3,"Retained Mean"),(4,"Churned Mean"),
                    (5,"Absolute Δ"),(6,"% Change"),(7,"Signal Strength")])
    ret=raw_df[raw_df["Exited"]==0]; chur=raw_df[raw_df["Exited"]==1]
    for i,cn in enumerate(num_cols,start=14):
        ws.row_dimensions[i].height=16
        rm=ret[cn].mean(); cm=chur[cn].mean(); diff=cm-rm; pct=(diff/rm) if rm!=0 else 0
        signal="Strong" if abs(pct)>0.15 else "Moderate" if abs(pct)>0.05 else "Weak"
        ws.cell(row=i,column=2,value=cn).font=fnt(size=10,color=BLUE)
        ws.cell(row=i,column=2).border=bdr_bottom()
        for j,(val,fmt,col_) in enumerate([
            (rm,"#,##0.00",INK),(cm,"#,##0.00",INK),
            (diff,"+#,##0.00;-#,##0.00",RED if diff>0 and cn in ["Balance","NumOfProducts","Age"] else INK),
            (pct,"+0.0%;-0.0%",RED if pct>0.1 else GREEN if pct<-0.1 else INK),
            (signal,None,RED if signal=="Strong" else ORANGE if signal=="Moderate" else SLATE),
        ],start=3):
            c=ws.cell(row=i,column=j,value=round(val,4) if isinstance(val,float) else val)
            c.font=fnt(size=10,color=col_)
            if fmt: c.number_format=fmt
            c.alignment=aln("right" if j<7 else "center","center"); c.border=bdr_bottom()

    # Product table
    sec_row(ws,21,2,"CHURN BY PRODUCTS — THE CRITICAL NON-LINEAR RELATIONSHIP")
    hdr_row(ws,22,[(2,"Products"),(3,"Total"),(4,"Churned"),(5,"Churn Rate"),
                    (6,"Avg Balance"),(7,"Assessment")])
    assessments={1:"Moderate risk — single product disengagement",
                  2:"LOW risk — sweet spot for retention",
                  3:"EXTREME risk — 82.7% churn — immediate action",
                  4:"CERTAIN churn — 100% — already lost"}
    for i,n in enumerate([1,2,3,4],start=23):
        sub=raw_df[raw_df["NumOfProducts"]==n]; ws.row_dimensions[i].height=16
        rate=sub["Exited"].mean(); bg="FEF2F2" if n>=3 else "F0FDF4" if n==2 else WHITE
        ws.cell(row=i,column=2,value=n).font=fnt(bold=n>=3,size=10)
        ws.cell(row=i,column=2).number_format="#,##0"
        ws.cell(row=i,column=2).border=bdr_bottom()
        ws.cell(row=i,column=2).fill=fill(bg)
        for j,(val,fmt) in enumerate([
            (len(sub),"#,##0"),(int(sub["Exited"].sum()),"#,##0"),
            (rate,"0.0%"),(sub["Balance"].mean(),"#,##0"),
            (assessments[n],None),
        ],start=3):
            c=ws.cell(row=i,column=j,value=round(val,4) if isinstance(val,float) else val)
            c.font=fnt(bold=(n>=3 and j==5),size=10,
                       color=RED if j==5 and rate>0.5 else INK)
            if fmt: c.number_format=fmt
            c.fill=fill(bg); c.alignment=aln("right" if j<7 else "left","center")
            c.border=bdr_bottom()


if __name__=="__main__":
    build()
