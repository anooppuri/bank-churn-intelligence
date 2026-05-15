"""
streamlit_app.py — Churn Intelligence System
Professional multi-page dashboard for bank customer churn prediction.
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from preprocessing import engineer_features, SCALE_COLS, CAT_COLS
from eda import get_eda_stats

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence | European Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DESIGN TOKENS ───────────────────────────────────────────────────────────
C = {
    "navy":    "#0D1B2A",
    "ink":     "#1B263B",
    "blue":    "#1B4F72",
    "sky":     "#2E86AB",
    "gold":    "#C9A84C",
    "amber":   "#E8A838",
    "cream":   "#F5F0E8",
    "white":   "#FFFFFF",
    "mist":    "#F0F4F8",
    "slate":   "#64748B",
    "steel":   "#94A3B8",
    "red":     "#B91C1C",
    "rose":    "#EF4444",
    "green":   "#15803D",
    "lime":    "#22C55E",
    "orange":  "#EA580C",
    "border":  "#E2E8F0",
}
RISK_HEX  = {"Critical Risk":"#B91C1C","High Risk":"#EA580C","Moderate Risk":"#D97706","Low Risk":"#15803D"}
RISK_BG   = {"Critical Risk":"#FEF2F2","High Risk":"#FFF7ED","Moderate Risk":"#FFFBEB","Low Risk":"#F0FDF4"}

def risk_band(p):
    return ("Critical Risk" if p>=0.65 else "High Risk" if p>=0.40
            else "Moderate Risk" if p>=0.20 else "Low Risk")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {C["navy"]} !important;
    border-right: 1px solid rgba(201,168,76,0.15);
}}
section[data-testid="stSidebar"] * {{
    color: {C["cream"]} !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.45rem 0.6rem !important;
    border-radius: 5px;
    transition: background 0.15s;
    cursor: pointer;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255,255,255,0.06) !important;
}}

/* ── Main ── */
.main .block-container {{
    padding: 1.8rem 2.2rem 3rem !important;
    max-width: 1280px !important;
}}

/* ── Page header ── */
.page-hero {{
    background: linear-gradient(135deg, {C["navy"]} 0%, {C["ink"]} 60%, {C["blue"]} 100%);
    border-radius: 10px;
    padding: 2rem 2.4rem 1.8rem;
    margin-bottom: 2rem;
    border-bottom: 3px solid {C["gold"]};
    position: relative;
    overflow: hidden;
}}
.page-hero::before {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(201,168,76,0.06);
    pointer-events: none;
}}
.page-hero h1 {{
    font-family: 'Playfair Display', serif !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: {C["white"]} !important;
    margin: 0 0 0.4rem !important;
    letter-spacing: -0.01em;
}}
.page-hero .subtitle {{
    color: {C["steel"]};
    font-size: 0.86rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}}
.page-hero .badge {{
    display: inline-block;
    background: rgba(201,168,76,0.18);
    border: 1px solid rgba(201,168,76,0.35);
    color: {C["amber"]};
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.7rem;
}}

/* ── KPI Cards ── */
.kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
.kpi-card {{
    flex: 1;
    background: {C["white"]};
    border: 1px solid {C["border"]};
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    border-top: 3px solid transparent;
    transition: box-shadow 0.2s;
}}
.kpi-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.07); }}
.kpi-card.blue  {{ border-top-color: {C["sky"]};    }}
.kpi-card.gold  {{ border-top-color: {C["gold"]};   }}
.kpi-card.red   {{ border-top-color: {C["red"]};    }}
.kpi-card.green {{ border-top-color: {C["green"]};  }}
.kpi-card.amber {{ border-top-color: {C["amber"]};  }}
.kpi-card .kpi-label {{
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: {C["slate"]}; margin-bottom: 0.4rem;
}}
.kpi-card .kpi-value {{
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem; font-weight: 500;
    color: {C["navy"]}; line-height: 1;
}}
.kpi-card .kpi-sub {{
    font-size: 0.73rem; color: {C["steel"]}; margin-top: 0.28rem;
}}
.kpi-card .kpi-delta {{
    font-size: 0.72rem; font-weight: 600;
    padding: 0.12rem 0.45rem; border-radius: 10px; margin-top: 0.3rem;
    display: inline-block;
}}
.delta-up   {{ background:#FEF2F2; color:{C["red"]};   }}
.delta-down {{ background:#F0FDF4; color:{C["green"]}; }}

/* ── Section labels ── */
.section-label {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.14em; color: {C["slate"]};
    padding-bottom: 0.55rem;
    border-bottom: 1px solid {C["border"]};
    margin-bottom: 1.1rem; margin-top: 0.5rem;
}}

/* ── Chart card ── */
.chart-card {{
    background: {C["white"]};
    border: 1px solid {C["border"]};
    border-radius: 8px;
    padding: 1.3rem;
}}

/* ── Risk badge ── */
.risk-pill {{
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem; font-weight: 500;
    letter-spacing: 0.04em;
    border: 1px solid transparent;
}}

/* ── Insight box ── */
.insight {{
    background: {C["mist"]};
    border-left: 3px solid {C["sky"]};
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: {C["ink"]};
    margin: 0.5rem 0;
    line-height: 1.55;
}}

/* ── Table ── */
.pro-table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
.pro-table th {{
    background: {C["navy"]}; color: {C["cream"]};
    padding: 0.6rem 0.9rem; text-align: left;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
}}
.pro-table td {{
    padding: 0.5rem 0.9rem; border-bottom: 1px solid {C["border"]};
    vertical-align: middle;
}}
.pro-table tr:hover td {{ background: {C["mist"]}; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0; background: {C["mist"]}; border-radius: 7px; padding: 3px;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.78rem !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: {C["slate"]} !important; border-radius: 5px;
}}
.stTabs [aria-selected="true"] {{
    background: {C["navy"]} !important;
    color: {C["white"]} !important;
}}

/* ── Prob bar ── */
.prob-bar-wrap {{
    background: {C["border"]}; border-radius: 4px; height: 8px; margin: 0.3rem 0;
}}
.prob-bar {{
    height: 8px; border-radius: 4px;
    transition: width 0.4s ease;
}}

/* Remove streamlit padding on number inputs */
div[data-testid="stNumberInput"] {{ margin-bottom: 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising Churn Intelligence System...")
def load_all():
    md  = os.path.join(BASE_DIR, "models")
    od  = os.path.join(BASE_DIR, "outputs")
    dd  = os.path.join(BASE_DIR, "data")

    with open(f"{md}/scaler.pkl","rb") as f:        scaler = pickle.load(f)
    with open(f"{md}/feature_names.json") as f:     feats  = json.load(f)
    with open(f"{od}/model_results.json") as f:     mres   = json.load(f)
    with open(f"{od}/summary_stats.json") as f:     summ   = json.load(f)
    with open(f"{md}/{summ['best_model'].replace(' ','_').lower()}.pkl","rb") as f:
        best_model = pickle.load(f)

    # Load all models for comparison
    all_models = {}
    for fname in os.listdir(md):
        if fname.endswith(".pkl") and fname not in ("scaler.pkl",):
            nm = fname.replace(".pkl","").replace("_"," ").title()
            try:
                with open(f"{md}/{fname}","rb") as f:
                    all_models[nm] = pickle.load(f)
            except Exception:
                pass

    raw     = pd.read_csv(f"{dd}/European_Bank.csv")
    scored  = pd.read_csv(f"{od}/scored_customers.csv")
    fi      = pd.read_csv(f"{od}/feature_importance.csv")
    shap_fi = pd.read_csv(f"{od}/shap_importance.csv")
    comp    = pd.read_csv(f"{od}/model_comparison.csv")

    # Pre-compute test predictions for ROC
    from preprocessing import preprocess
    from sklearn.model_selection import train_test_split
    X, y, _, _ = preprocess(raw, scaler=scaler, fit_scaler=False)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

    return dict(
        scaler=scaler, feats=feats, mres=mres, summ=summ,
        best_model=best_model, all_models=all_models,
        raw=raw, scored=scored, fi=fi, shap_fi=shap_fi, comp=comp,
        X_test=X_test, y_test=y_test
    )


def predict_single(model, scaler, feats, d: dict) -> float:
    try:
        df = pd.DataFrame([d])
        df = engineer_features(df)
        df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
        for c in feats:
            if c not in df.columns: df[c] = 0
        df = df[feats]
        sc = [c for c in SCALE_COLS if c in df.columns]
        df[sc] = scaler.transform(df[sc])
        return float(model.predict_proba(df.values)[0][1])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.5


# ─── CHART HELPERS ────────────────────────────────────────────────────────────
def chart_layout(fig, height=320, margin=None, title=None):
    m = margin or dict(t=40 if title else 15, b=15, l=15, r=15)
    fig.update_layout(
        height=height, margin=m,
        plot_bgcolor=C["white"], paper_bgcolor=C["white"],
        font=dict(family="DM Sans", size=11, color=C["ink"]),
        title=dict(text=title, font=dict(size=13, color=C["navy"]),
                   x=0, xanchor="left") if title else None,
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def gauge_chart(prob):
    rb  = risk_band(prob)
    col = RISK_HEX[rb]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        number=dict(suffix="%", font=dict(size=38, family="DM Mono", color=C["navy"])),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=0, tickcolor=C["border"],
                      tickfont=dict(size=9, color=C["slate"])),
            bar=dict(color=col, thickness=0.68),
            bgcolor=C["white"], borderwidth=0,
            steps=[
                dict(range=[0,20],  color="#DCFCE7"),
                dict(range=[20,40], color="#FEF9C3"),
                dict(range=[40,65], color="#FFEDD5"),
                dict(range=[65,100],color="#FEE2E2"),
            ],
            threshold=dict(
                line=dict(color=C["navy"], width=2),
                thickness=0.78, value=prob*100
            )
        )
    ))
    fig.update_layout(
        height=200, margin=dict(t=10,b=5,l=20,r=20),
        paper_bgcolor=C["white"], plot_bgcolor=C["white"],
        font=dict(family="DM Sans")
    )
    return fig


def kpi(label, value, sub="", variant="blue", delta=None, delta_dir=None):
    delta_html = ""
    if delta is not None:
        cls = "delta-up" if delta_dir=="bad" else "delta-down"
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def insight(html):
    st.markdown(f'<div class="insight">{html}</div>', unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1.4rem 0.6rem 1rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        color:{C['gold']};letter-spacing:0.14em;
                        text-transform:uppercase;margin-bottom:0.3rem;">
                European Bank
            </div>
            <div style="font-family:'Playfair Display',serif;font-size:1.25rem;
                        color:{C['white']};font-weight:700;line-height:1.2;">
                Churn<br>Intelligence
            </div>
            <div style="margin-top:0.5rem;height:2px;
                        background:linear-gradient(90deg,{C['gold']},transparent);">
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "",
            options=[
                "🏠  Executive Dashboard",
                "🔬  EDA & Analytics",
                "🎯  Risk Calculator",
                "📈  Model Performance",
                "🔍  Feature Intelligence",
                "🔬  What-If Simulator",
                "📋  Customer Register",
            ],
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:0.9rem 0.7rem;background:rgba(201,168,76,0.1);
                    border:1px solid rgba(201,168,76,0.2);border-radius:7px;
                    font-size:0.75rem;color:{C['steel']};">
            <div style="color:{C['gold']};font-weight:600;font-size:0.7rem;
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">
                Model Status
            </div>
            <div>✓ &nbsp;Gradient Boosting</div>
            <div>✓ &nbsp;SMOTE Rebalanced</div>
            <div>✓ &nbsp;Isotonic Calibrated</div>
            <div>✓ &nbsp;SHAP Explained</div>
            <div style="margin-top:0.6rem;padding-top:0.5rem;
                        border-top:1px solid rgba(255,255,255,0.08);
                        font-family:'DM Mono',monospace;color:{C['amber']};">
                AUC · 0.8671
            </div>
        </div>
        """, unsafe_allow_html=True)

    return page.split("  ", 1)[-1].strip()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(A):
    summ   = A["summ"]
    scored = A["scored"]
    raw    = A["raw"].copy()

    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">Executive Summary</div>
        <h1>Churn Intelligence System</h1>
        <p class="subtitle">
            Predictive Risk Scoring &amp; Retention Analytics &nbsp;·&nbsp;
            European Retail Bank Portfolio &nbsp;·&nbsp;
            10,000 Customers &nbsp;·&nbsp; Gradient Boosting + SMOTE + Isotonic Calibration
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi("Total Customers", f"{summ['total_customers']:,}", "Active portfolio", "blue")
    with k2: kpi("Actual Churn Rate", f"{summ['actual_churn_rate']:.1%}",
                  f"{summ['actual_churners']:,} customers lost", "red",
                  delta="↑ High", delta_dir="bad")
    with k3: kpi("Model ROC-AUC", f"{summ['best_roc_auc']:.4f}",
                  summ["best_model"], "gold")
    with k4: kpi("Recall", f"{summ['best_recall']:.1%}",
                  "Churners correctly flagged", "amber")
    with k5: kpi("Precision", f"{summ['best_precision']:.1%}",
                  "Flag accuracy (↑ +5pp vs v1)", "green")

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk distribution + probability histogram
    section("PORTFOLIO RISK SEGMENTATION")
    col_a, col_b = st.columns([1.15, 1])

    with col_a:
        order = ["Critical Risk","High Risk","Moderate Risk","Low Risk"]
        rb = (scored["RiskBand"].value_counts()
              .reindex(order).reset_index())
        rb.columns = ["Band","Count"]
        rb["Pct"] = rb["Count"] / rb["Count"].sum()

        fig = go.Figure()
        for _, r in rb.iterrows():
            fig.add_trace(go.Bar(
                x=[r["Band"]], y=[r["Count"]],
                marker_color=RISK_HEX[r["Band"]],
                marker_opacity=0.88,
                text=f"{r['Count']:,}<br><span style='font-size:10px'>{r['Pct']:.1%}</span>",
                textposition="outside",
                textfont=dict(family="DM Mono", size=10),
                showlegend=False,
                name=r["Band"]
            ))
        chart_layout(fig, height=300, title="Customer Risk Band Distribution")
        fig.update_layout(showlegend=False,
                          yaxis=dict(title="Customers", gridcolor=C["border"]),
                          xaxis=dict(categoryorder="array", categoryarray=order))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        for exited, label, color in [(0,"Retained",C["green"]),(1,"Churned",C["red"])]:
            fig2.add_trace(go.Histogram(
                x=scored[scored["Exited"]==exited]["ChurnProbability"],
                name=label, marker_color=color, opacity=0.72, nbinsx=40,
                hovertemplate=f"{label}<br>Probability: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
            ))
        chart_layout(fig2, height=300, title="Predicted Probability Distribution")
        fig2.update_layout(barmode="overlay",
                           xaxis=dict(title="Churn Probability"),
                           yaxis=dict(title="Count"),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    # Segment breakdown — 3 columns
    section("CHURN RATE BY SEGMENT")
    c1, c2, c3 = st.columns(3)

    with c1:
        geo = raw.groupby("Geography")["Exited"].mean().reset_index()
        geo.columns = ["Geography","Rate"]
        geo["Color"] = geo["Rate"].apply(
            lambda v: C["red"] if v>0.25 else C["amber"] if v>0.17 else C["green"])
        fig3 = go.Figure(go.Bar(
            x=geo["Geography"], y=geo["Rate"],
            marker_color=geo["Color"].tolist(), marker_opacity=0.85,
            text=[f"{v:.1%}" for v in geo["Rate"]],
            textposition="outside", textfont=dict(family="DM Mono", size=11)
        ))
        chart_layout(fig3, height=250, title="By Geography")
        fig3.update_layout(yaxis=dict(tickformat=".0%", title="Churn Rate"))
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        prod = raw.groupby("NumOfProducts")["Exited"].mean().reset_index()
        prod.columns = ["Products","Rate"]
        prod["Color"] = prod["Rate"].apply(
            lambda v: C["red"] if v>0.5 else C["green"])
        fig4 = go.Figure(go.Bar(
            x=prod["Products"].astype(str), y=prod["Rate"],
            marker_color=prod["Color"].tolist(), marker_opacity=0.85,
            text=[f"{v:.1%}" for v in prod["Rate"]],
            textposition="outside", textfont=dict(family="DM Mono", size=11)
        ))
        chart_layout(fig4, height=250, title="By Products Held")
        fig4.update_layout(yaxis=dict(tickformat=".0%", title="Churn Rate"),
                           xaxis=dict(title="Number of Products"))
        st.plotly_chart(fig4, use_container_width=True)

    with c3:
        raw["AgeGroup"] = pd.cut(raw["Age"], bins=[17,30,40,50,60,100],
                                  labels=["18–30","31–40","41–50","51–60","60+"])
        age = raw.groupby("AgeGroup", observed=True)["Exited"].mean().reset_index()
        age.columns = ["Group","Rate"]
        age["Color"] = age["Rate"].apply(
            lambda v: C["red"] if v>0.38 else C["amber"] if v>0.22 else C["green"])
        fig5 = go.Figure(go.Bar(
            x=age["Group"].astype(str), y=age["Rate"],
            marker_color=age["Color"].tolist(), marker_opacity=0.85,
            text=[f"{v:.1%}" for v in age["Rate"]],
            textposition="outside", textfont=dict(family="DM Mono", size=11)
        ))
        chart_layout(fig5, height=250, title="By Age Group")
        fig5.update_layout(yaxis=dict(tickformat=".0%", title="Churn Rate"),
                           xaxis=dict(title="Age Band"))
        st.plotly_chart(fig5, use_container_width=True)

    # Key insights
    section("KEY ANALYTICAL FINDINGS")
    i1, i2, i3 = st.columns(3)
    with i1: insight("""
        <b>🇩🇪 Germany Risk Premium (2×)</b><br>
        German customers exit at <b>32.4%</b> vs 16.2% in France and 16.7% in Spain.
        Priority market for targeted retention budget allocation.
    """)
    with i2: insight("""
        <b>📦 The 3-Product Cliff</b><br>
        Customers with 3 products churn at <b>82.7%</b>; those with 4 at <b>100%</b>.
        Product overloading is the single most actionable churn predictor.
    """)
    with i3: insight("""
        <b>👤 Age-44 Inflection Point</b><br>
        Mean churner age: <b>44.8 yrs</b> vs 37.4 for retained customers.
        This cohort holds 25% more balance — highest CLV at risk per departure.
    """)

    # Active member impact
    i4, i5, i6 = st.columns(3)
    with i4: insight("""
        <b>⚡ Activity Multiplier</b><br>
        Inactive members churn at <b>26.9%</b> vs <b>14.3%</b> for active — a 1.9×
        ratio making engagement re-activation the highest-ROI retention lever.
    """)
    with i5: insight("""
        <b>💰 Paradox of Wealth</b><br>
        Churners carry mean balance of <b>€91,108</b> vs €72,745 for retained.
        High-balance customers have higher expectations — premium servicing is warranted.
    """)
    with i6: insight("""
        <b>♀ Gender Differential</b><br>
        Female customers churn at <b>25.1%</b> vs 16.5% male — a gap worth
        exploring for product design and communications personalisation.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
def page_eda(A):
    raw = A["raw"].copy()

    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">Exploratory Data Analysis</div>
        <h1>Statistical Profile &amp; Distribution Analysis</h1>
        <p class="subtitle">
            10,000 customers · 14 raw features · 7 engineered features · No missing values
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_num, tab_cat, tab_cross, tab_dist = st.tabs([
        "📊  Numeric Features",
        "🏷  Categorical Breakdown",
        "🔗  Cross-Tabulations",
        "📉  Distributions"
    ])

    # ── Numeric summary ──────────────────────────────────────────────────────
    with tab_num:
        section("DESCRIPTIVE STATISTICS — CHURNED VS RETAINED")
        num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]

        ret  = raw[raw["Exited"]==0]
        chur = raw[raw["Exited"]==1]

        rows = []
        for c in num_cols:
            rows.append({
                "Feature":          c,
                "Overall Mean":     raw[c].mean(),
                "Retained Mean":    ret[c].mean(),
                "Churned Mean":     chur[c].mean(),
                "Δ (Ch−Re)":        chur[c].mean() - ret[c].mean(),
                "% Change":         (chur[c].mean() - ret[c].mean()) / ret[c].mean(),
                "Std Dev":          raw[c].std(),
                "Min":              raw[c].min(),
                "Median":           raw[c].median(),
                "Max":              raw[c].max(),
            })
        stats_df = pd.DataFrame(rows)

        # Color the delta column
        def color_delta(val):
            if isinstance(val, float):
                if val > 0: return "color: #B91C1C; font-weight:600"
                if val < 0: return "color: #15803D; font-weight:600"
            return ""

        fmt = {
            "Overall Mean": "{:,.1f}", "Retained Mean": "{:,.1f}",
            "Churned Mean": "{:,.1f}", "Δ (Ch−Re)": "{:+,.1f}",
            "% Change": "{:+.1%}", "Std Dev": "{:,.1f}",
            "Min": "{:,.0f}", "Median": "{:,.1f}", "Max": "{:,.0f}",
        }
        st.dataframe(
            stats_df.style
                .format(fmt)
                .map(color_delta, subset=["Δ (Ch−Re)","% Change"]),
            use_container_width=True, height=280, hide_index=True
        )

        # Box plots
        section("DISTRIBUTION BY CHURN STATUS")
        bc1, bc2, bc3 = st.columns(3)
        box_pairs = [
            (bc1, "Age", "Age Distribution"),
            (bc2, "Balance", "Account Balance (€)"),
            (bc3, "CreditScore", "Credit Score"),
        ]
        for col, feat, title in box_pairs:
            with col:
                fig = go.Figure()
                for exited, label, color in [(0,"Retained",C["green"]),(1,"Churned",C["red"])]:
                    fig.add_trace(go.Box(
                        y=raw[raw["Exited"]==exited][feat],
                        name=label, marker_color=color, boxmean="sd",
                        line_color=color, fillcolor="rgba(21,128,61,0.15)" if color=="#15803D" else "rgba(185,28,28,0.15)"
                    ))
                chart_layout(fig, height=280, title=title)
                st.plotly_chart(fig, use_container_width=True)

        bc4, bc5, bc6 = st.columns(3)
        box_pairs2 = [
            (bc4, "Tenure", "Tenure (Years)"),
            (bc5, "EstimatedSalary", "Estimated Salary (€)"),
            (bc6, "NumOfProducts", "Number of Products"),
        ]
        for col, feat, title in box_pairs2:
            with col:
                fig = go.Figure()
                for exited, label, color in [(0,"Retained",C["green"]),(1,"Churned",C["red"])]:
                    fig.add_trace(go.Box(
                        y=raw[raw["Exited"]==exited][feat],
                        name=label, marker_color=color, boxmean="sd",
                        line_color=color, fillcolor="rgba(21,128,61,0.15)" if color=="#15803D" else "rgba(185,28,28,0.15)"
                    ))
                chart_layout(fig, height=280, title=title)
                st.plotly_chart(fig, use_container_width=True)

    # ── Categorical ──────────────────────────────────────────────────────────
    with tab_cat:
        section("CATEGORICAL FEATURES — CHURN RATE BREAKDOWN")
        cat_data = [
            ("Geography",      raw.groupby("Geography")["Exited"].agg(["mean","sum","count"]).reset_index()),
            ("Gender",         raw.groupby("Gender")["Exited"].agg(["mean","sum","count"]).reset_index()),
            ("NumOfProducts",  raw.groupby("NumOfProducts")["Exited"].agg(["mean","sum","count"]).reset_index()),
            ("HasCrCard",      raw.groupby("HasCrCard")["Exited"].agg(["mean","sum","count"]).reset_index()),
            ("IsActiveMember", raw.groupby("IsActiveMember")["Exited"].agg(["mean","sum","count"]).reset_index()),
        ]
        cc1, cc2, cc3 = st.columns(3)
        col_cycle = [cc1, cc2, cc3, cc1, cc2]

        for i, (feat, grp) in enumerate(cat_data):
            grp.columns = [feat, "Churn Rate","Churned","Total"]
            with col_cycle[i]:
                fig = go.Figure(go.Bar(
                    x=grp[feat].astype(str), y=grp["Churn Rate"],
                    marker_color=[C["red"] if v>0.3 else C["amber"] if v>0.2 else C["green"]
                                  for v in grp["Churn Rate"]],
                    text=[f"{v:.1%}" for v in grp["Churn Rate"]],
                    textposition="outside", textfont=dict(family="DM Mono",size=11)
                ))
                chart_layout(fig, height=240, title=f"Churn by {feat}")
                fig.update_layout(yaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig, use_container_width=True)

    # ── Cross-tabulations ────────────────────────────────────────────────────
    with tab_cross:
        section("CROSS-TABULATION — CHURN RATE HEATMAP")
        cr1, cr2 = st.columns(2)

        with cr1:
            # Geography × Gender
            pivot = raw.pivot_table(values="Exited", index="Geography",
                                     columns="Gender", aggfunc="mean")
            fig_h = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0,"#F0FDF4"],[0.5,"#FEF9C3"],[1,"#FEE2E2"]],
                text=[[f"{v:.1%}" for v in row] for row in pivot.values],
                texttemplate="%{text}", textfont=dict(family="DM Mono",size=13),
                showscale=True, zmin=0.10, zmax=0.40,
                colorbar=dict(tickformat=".0%", thickness=12)
            ))
            chart_layout(fig_h, height=280, title="Churn Rate: Geography × Gender")
            st.plotly_chart(fig_h, use_container_width=True)

        with cr2:
            # Geography × NumOfProducts
            pivot2 = raw.pivot_table(values="Exited", index="Geography",
                                      columns="NumOfProducts", aggfunc="mean")
            fig_h2 = go.Figure(go.Heatmap(
                z=pivot2.values, x=[str(c) for c in pivot2.columns],
                y=pivot2.index.tolist(),
                colorscale=[[0,"#F0FDF4"],[0.4,"#FEF9C3"],[1,"#FEE2E2"]],
                text=[[f"{v:.1%}" if not np.isnan(v) else "N/A"
                       for v in row] for row in pivot2.values],
                texttemplate="%{text}", textfont=dict(family="DM Mono",size=12),
                showscale=True, zmin=0, zmax=1.0,
                colorbar=dict(tickformat=".0%", thickness=12)
            ))
            chart_layout(fig_h2, height=280,
                         title="Churn Rate: Geography × Num Products")
            fig_h2.update_layout(xaxis=dict(title="Number of Products"))
            st.plotly_chart(fig_h2, use_container_width=True)

        # Active × Age group
        section("ACTIVE MEMBER STATUS × AGE GROUP")
        raw["AgeGroup"] = pd.cut(raw["Age"], bins=[17,30,40,50,60,100],
                                  labels=["18–30","31–40","41–50","51–60","60+"])
        pivot3 = raw.pivot_table(values="Exited", index="IsActiveMember",
                                  columns="AgeGroup", aggfunc="mean", observed=True)
        pivot3.index = ["Inactive","Active"]
        fig_h3 = go.Figure(go.Heatmap(
            z=pivot3.values, x=pivot3.columns.astype(str).tolist(),
            y=pivot3.index.tolist(),
            colorscale=[[0,"#F0FDF4"],[0.5,"#FEF9C3"],[1,"#FEE2E2"]],
            text=[[f"{v:.1%}" if not np.isnan(v) else "N/A" for v in row]
                  for row in pivot3.values],
            texttemplate="%{text}", textfont=dict(family="DM Mono",size=13),
            showscale=True, zmin=0.05, zmax=0.65,
            colorbar=dict(tickformat=".0%", thickness=12)
        ))
        chart_layout(fig_h3, height=200, title="Churn Rate: Activity Status × Age Group")
        st.plotly_chart(fig_h3, use_container_width=True)

    # ── Distributions ────────────────────────────────────────────────────────
    with tab_dist:
        section("PROBABILITY DENSITY — KEY FEATURES")
        feat_sel = st.selectbox("Select feature", [
            "Age","Balance","CreditScore","Tenure","EstimatedSalary"
        ], key="eda_dist")

        figD = go.Figure()
        for exited, label, color in [(0,"Retained",C["green"]),(1,"Churned",C["red"])]:
            vals = raw[raw["Exited"]==exited][feat_sel]
            figD.add_trace(go.Histogram(
                x=vals, name=label, nbinsx=50,
                marker_color=color, opacity=0.65,
                histnorm="probability density"
            ))
        chart_layout(figD, height=340, title=f"{feat_sel} — Probability Density")
        figD.update_layout(barmode="overlay",
                           xaxis=dict(title=feat_sel),
                           yaxis=dict(title="Density"),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figD, use_container_width=True)

        # Scatter
        section("BIVARIATE SCATTER — AGE vs BALANCE")
        samp = raw.sample(min(2000, len(raw)), random_state=42)
        figS = px.scatter(
            samp, x="Age", y="Balance",
            color=samp["Exited"].map({0:"Retained",1:"Churned"}),
            color_discrete_map={"Retained":C["green"],"Churned":C["red"]},
            opacity=0.5, size_max=6,
            hover_data=["Geography","Gender","NumOfProducts","CreditScore"]
        )
        chart_layout(figS, height=360)
        figS.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(figS, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def page_calculator(A):
    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">Individual Assessment</div>
        <h1>Customer Churn Risk Calculator</h1>
        <p class="subtitle">
            Input customer attributes to generate a calibrated churn probability score
            with automated risk banding and recommended retention action.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_score = st.columns([1, 1])

    with col_form:
        section("CUSTOMER PROFILE INPUT")
        f1, f2 = st.columns(2)
        with f1:
            credit  = st.slider("Credit Score",    300, 850, 650, 10, key="c_cr")
            age     = st.slider("Age",              18,  92,  40,  1,  key="c_ag")
            tenure  = st.slider("Tenure (years)",   0,   10,  5,   1,  key="c_tn")
            n_prod  = st.selectbox("No. of Products",[1,2,3,4], index=1, key="c_np")
        with f2:
            balance = st.number_input("Balance (€)",  0.0, 300000.0, 80000.0, 1000.0, key="c_bl")
            salary  = st.number_input("Est. Salary (€)", 0.0, 250000.0, 60000.0, 1000.0, key="c_sl")
            cc      = st.selectbox("Credit Card", ["Yes","No"], key="c_cc")
            active  = st.selectbox("Active Member",["Yes","No"], key="c_ac")
        geo    = st.selectbox("Geography", ["France","Germany","Spain"], key="c_ge")
        gender = st.selectbox("Gender",    ["Female","Male"],           key="c_gn")

        customer = dict(
            CreditScore=credit, Age=age, Tenure=tenure, Balance=balance,
            NumOfProducts=n_prod, HasCrCard=1 if cc=="Yes" else 0,
            IsActiveMember=1 if active=="Yes" else 0,
            EstimatedSalary=salary, Geography=geo, Gender=gender
        )

    with col_score:
        prob = predict_single(A["best_model"], A["scaler"], A["feats"], customer)
        rb   = risk_band(prob)
        rhex = RISK_HEX[rb]
        rbg  = RISK_BG[rb]

        section("RISK ASSESSMENT")
        st.plotly_chart(gauge_chart(prob), use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center; margin:-0.5rem 0 1rem;">
            <span class="risk-pill"
                  style="background:{rbg};color:{rhex};border-color:{rhex}44;">
                {rb}
            </span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1: kpi("Churn Probability", f"{prob:.1%}", "Calibrated score",
                      "red" if prob>0.5 else "green")
        with m2: kpi("Retention Score",   f"{1-prob:.1%}", "Inverse probability",
                      "green" if (1-prob)>0.6 else "amber")

        st.markdown("<br>", unsafe_allow_html=True)
        if   prob >= 0.65:
            action = "🚨 <b>Immediate Retention Action</b> — Assign dedicated relationship manager. Deliver personalised retention package within 48 hours. Escalate to branch director."
        elif prob >= 0.40:
            action = "⚠️ <b>Proactive Engagement</b> — Schedule outreach call within 2 weeks. Review product fit and introduce engagement incentive."
        elif prob >= 0.20:
            action = "📋 <b>Monitor &amp; Nurture</b> — Include in next quarterly campaign. Review for cross-sell opportunity and engagement activation."
        else:
            action = "✅ <b>Stable Customer</b> — Standard relationship management. Consider as brand advocate for referral programme."
        insight(action)

    # Probability bar for top features
    st.markdown("<br>", unsafe_allow_html=True)
    section("GLOBAL MODEL DRIVERS (SHAP)")
    shap_top = A["shap_fi"].head(10)
    max_s = shap_top["SHAP_Importance"].max()
    html_rows = ""
    for _, row in shap_top.iterrows():
        pct = (row["SHAP_Importance"] / max_s) * 100
        html_rows += f"""
        <tr>
            <td style="width:30%;font-size:0.82rem;color:{C['ink']};padding:0.45rem 0.9rem;">
                {row['Feature']}
            </td>
            <td style="padding:0.45rem 0.9rem;">
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{pct:.1f}%;background:{C['sky']};"></div>
                </div>
            </td>
            <td style="width:15%;font-family:'DM Mono',monospace;font-size:0.8rem;
                       text-align:right;padding:0.45rem 0.9rem;color:{C['slate']};">
                {row['SHAP_Importance']:.4f}
            </td>
        </tr>"""
    st.markdown(f"""
    <table class="pro-table">
        <thead><tr>
            <th>Feature</th><th>SHAP Importance</th><th>Score</th>
        </tr></thead>
        <tbody>{html_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_models(A):
    comp    = A["comp"]
    mres    = A["mres"]
    summ    = A["summ"]
    X_test  = A["X_test"]
    y_test  = A["y_test"]
    models  = A["all_models"]

    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">Model Evaluation</div>
        <h1>ML Model Performance Analysis</h1>
        <p class="subtitle">
            5 algorithms evaluated · Stratified 80/10/10 split ·
            Youden's J threshold · Isotonic probability calibration
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics table
    section("COMPARATIVE PERFORMANCE METRICS")
    fmt_cols = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC",
                "Avg Precision","Specificity"]
    st.dataframe(
        comp.style
            .apply(lambda col: [
                f"background-color: rgba(46,134,171,{0.1 + 0.5*((v-0.45)/0.47):.2f})"
                if col.name in ["ROC-AUC","F1 Score","Recall"] and isinstance(v,float)
                else f"background-color: rgba(21,128,61,{0.1 + 0.5*((v-0.40)/0.30):.2f})"
                if col.name == "Precision" and isinstance(v,float)
                else ""
                for v in col
            ], axis=0)
            .format({c:"{:.4f}" for c in fmt_cols})
            .apply(lambda row: [
                "font-weight:700; background:#FFF8E7" if row["Model"]==summ["best_model"]
                else "" for _ in row
            ], axis=1),
        use_container_width=True, height=230, hide_index=True
    )

    tab_roc, tab_radar, tab_cm, tab_pr = st.tabs([
        "📈  ROC Curves",
        "🕸  Radar Chart",
        "🧮  Confusion Matrices",
        "🎯  Precision-Recall"
    ])

    model_colors = [C["sky"], C["gold"], C["red"], C["green"], C["slate"]]

    # ── ROC Curves ───────────────────────────────────────────────────────────
    with tab_roc:
        section("ROC CURVES — ALL MODELS")
        fig_roc = go.Figure()
        # Diagonal
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color=C["border"], dash="dot", width=1.5),
            name="Random Classifier", showlegend=True
        ))
        nm_map = {
            "Gradient Boosting": "gradient_boosting",
            "Xgboost": "xgboost",
            "Random Forest": "random_forest",
            "Decision Tree": "decision_tree",
            "Logistic Regression": "logistic_regression",
        }
        for i, (_, row) in enumerate(comp.iterrows()):
            mname = row["Model"]
            key   = nm_map.get(mname, mname.replace(" ","_").lower())
            model = models.get(key) or models.get(mname)
            if model is None:
                continue
            try:
                probs = model.predict_proba(X_test)[:,1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc  = row["ROC-AUC"]
                lw   = 2.5 if mname==summ["best_model"] else 1.8
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{mname} (AUC={auc:.4f})",
                    line=dict(color=model_colors[i%len(model_colors)], width=lw),
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
                ))
            except Exception:
                pass
        chart_layout(fig_roc, height=420,
                     title="Receiver Operating Characteristic — All Models")
        fig_roc.update_layout(
            xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02,1.02]),
            yaxis=dict(title="True Positive Rate (Recall)", range=[-0.02,1.02]),
            legend=dict(orientation="v", x=0.55, y=0.08,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=C["border"], borderwidth=1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Radar ────────────────────────────────────────────────────────────────
    with tab_radar:
        section("MULTI-METRIC RADAR — MODEL COMPARISON")
        cats = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","Specificity"]
        fig_r = go.Figure()
        for i, (_, row) in enumerate(comp.iterrows()):
            vals = [row[c] for c in cats] + [row[cats[0]]]
            lw   = 2.5 if row["Model"]==summ["best_model"] else 1.5
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats+[cats[0]], name=row["Model"],
                line=dict(color=model_colors[i%len(model_colors)], width=lw),
                fill="toself",
                fillcolor=model_colors[i%len(model_colors)],
                opacity=0.10 if row["Model"]!=summ["best_model"] else 0.18
            ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.55,0.95],
                                gridcolor=C["border"],
                                tickfont=dict(size=8, color=C["slate"])),
                angularaxis=dict(gridcolor=C["border"])
            ),
            height=440, margin=dict(t=60,b=30,l=30,r=30),
            paper_bgcolor=C["white"],
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            font=dict(family="DM Sans"),
            title=dict(text=f"Model Performance Radar (Best: {summ['best_model']})",
                       font=dict(size=13, color=C["navy"]), x=0)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    with tab_cm:
        section("CONFUSION MATRICES AT OPTIMAL THRESHOLD")
        cm_cols = st.columns(min(3, len(comp)))
        nm_list = comp["Model"].tolist()

        for i, mname in enumerate(nm_list[:5]):
            key   = nm_map.get(mname, mname.replace(" ","_").lower())
            model = models.get(key) or models.get(mname)
            col   = cm_cols[i % 3]
            if model is None:
                continue
            with col:
                try:
                    probs = model.predict_proba(X_test)[:,1]
                    thresh = mres[mname]["optimal_threshold"]
                    preds  = (probs >= thresh).astype(int)
                    cm     = confusion_matrix(y_test, preds)
                    labels = ["Retained","Churned"]
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm, x=labels, y=labels,
                        colorscale=[[0,C["mist"]],[1,C["navy"]]],
                        showscale=False,
                        text=cm.tolist(),
                        texttemplate="<b>%{text}</b>",
                        textfont=dict(family="DM Mono", size=18)
                    ))
                    auc = mres[mname]["roc_auc"]
                    chart_layout(fig_cm, height=240,
                                 title=f"{mname[:18]}<br><span style='font-size:11px;color:{C['slate']};'>AUC {auc:.4f} · Thresh {thresh:.2f}</span>")
                    fig_cm.update_layout(
                        xaxis=dict(title="Predicted", tickfont=dict(size=10)),
                        yaxis=dict(title="Actual", tickfont=dict(size=10))
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception:
                    pass

    # ── Precision-Recall ─────────────────────────────────────────────────────
    with tab_pr:
        section("PRECISION-RECALL CURVES — CLASS IMBALANCE ROBUST")
        fig_pr = go.Figure()
        baseline_pr = y_test.mean()
        fig_pr.add_trace(go.Scatter(
            x=[0,1], y=[baseline_pr,baseline_pr], mode="lines",
            line=dict(color=C["border"],dash="dot",width=1.5),
            name=f"Baseline ({baseline_pr:.2%})"
        ))
        for i, (_, row) in enumerate(comp.iterrows()):
            mname = row["Model"]
            key   = nm_map.get(mname, mname.replace(" ","_").lower())
            model = models.get(key) or models.get(mname)
            if model is None: continue
            try:
                probs = model.predict_proba(X_test)[:,1]
                prec, rec, _ = precision_recall_curve(y_test, probs)
                ap = row["Avg Precision"]
                lw = 2.5 if mname==summ["best_model"] else 1.8
                fig_pr.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines",
                    name=f"{mname} (AP={ap:.4f})",
                    line=dict(color=model_colors[i%len(model_colors)],width=lw)
                ))
            except Exception:
                pass
        chart_layout(fig_pr, height=420,
                     title="Precision-Recall Curves — All Models")
        fig_pr.update_layout(
            xaxis=dict(title="Recall (True Positive Rate)", range=[-0.02,1.02]),
            yaxis=dict(title="Precision", range=[-0.02,1.05]),
            legend=dict(orientation="v", x=0.62, y=0.95,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=C["border"], borderwidth=1)
        )
        st.plotly_chart(fig_pr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
def page_features(A):
    fi      = A["fi"]
    shap_fi = A["shap_fi"]
    raw     = A["raw"]

    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">Feature Intelligence</div>
        <h1>Feature Importance &amp; SHAP Analysis</h1>
        <p class="subtitle">
            21 features (14 raw + 7 engineered) · Tree importance vs SHAP mean |φ| ·
            Pearson correlation analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_fi, tab_shap, tab_corr, tab_eng = st.tabs([
        "🌲  Tree Importance",
        "🔦  SHAP Values",
        "📐  Correlations",
        "⚙️  Engineered Features"
    ])

    with tab_fi:
        section("GRADIENT BOOSTING — FEATURE IMPORTANCE RANKING")
        top_n = st.slider("Show top N features", 5, len(fi), 15, 1, key="fi_n")
        fi_top = fi.head(top_n).sort_values("Importance")

        cum = fi_top["Importance"].cumsum() / fi["Importance"].sum()
        fig_fi = make_subplots(specs=[[{"secondary_y":True}]])
        fig_fi.add_trace(go.Bar(
            x=fi_top["Importance"], y=fi_top["Feature"],
            orientation="h",
            marker=dict(
                color=fi_top["Importance"],
                colorscale=[[0,C["mist"]],[0.5,C["sky"]],[1,C["navy"]]],
                showscale=False
            ),
            text=[f"{v:.4f}" for v in fi_top["Importance"]],
            textposition="outside", textfont=dict(family="DM Mono",size=9),
            name="Importance"
        ), secondary_y=False)
        chart_layout(fig_fi, height=500 if top_n>10 else 360,
                     title=f"Top {top_n} Feature Importances")
        fig_fi.update_layout(
            yaxis=dict(title=""),
            xaxis=dict(title="Importance Score", gridcolor=C["border"]),
            showlegend=False
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Cumulative importance
        cum_df = fi.copy()
        cum_df["CumImportance"] = cum_df["Importance"].cumsum() / cum_df["Importance"].sum()
        cum_df["Rank"] = range(1, len(cum_df)+1)
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum_df["Rank"], y=cum_df["CumImportance"],
            mode="lines+markers",
            line=dict(color=C["sky"],width=2.5),
            marker=dict(size=6, color=C["navy"]),
            fill="tozeroy", fillcolor="rgba(46,134,171,0.13)",
            name="Cumulative Importance"
        ))
        fig_cum.add_hline(y=0.8, line_dash="dash", line_color=C["amber"],
                          annotation_text="80% threshold",
                          annotation_position="right")
        chart_layout(fig_cum, height=280,
                     title="Cumulative Feature Importance")
        fig_cum.update_layout(
            xaxis=dict(title="Feature Rank"),
            yaxis=dict(title="Cumulative Importance", tickformat=".0%")
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab_shap:
        section("SHAP MEAN |φ| — MODEL-AGNOSTIC IMPORTANCE")
        shap_top = shap_fi.head(top_n if "fi_n" in st.session_state else 15)
        shap_top = shap_top.sort_values("SHAP_Importance")
        max_s = shap_top["SHAP_Importance"].max()
        alphas = [0.3 + 0.7*(v/max_s) for v in shap_top["SHAP_Importance"]]
        bar_cols = [f"rgba(185,28,28,{a:.2f})" for a in alphas]
        fig_s = go.Figure(go.Bar(
            x=shap_top["SHAP_Importance"], y=shap_top["Feature"],
            orientation="h", marker_color=bar_cols,
            text=[f"{v:.4f}" for v in shap_top["SHAP_Importance"]],
            textposition="outside", textfont=dict(family="DM Mono",size=9)
        ))
        chart_layout(fig_s, height=500 if len(shap_top)>10 else 360,
                     title="SHAP Importance — Mean Absolute Shapley Values")
        fig_s.update_layout(
            xaxis=dict(title="Mean |SHAP Value|"),
            yaxis=dict(title=""),
            showlegend=False
        )
        st.plotly_chart(fig_s, use_container_width=True)

        insight("""
            <b>SHAP vs Tree Importance:</b> Tree-based importance can overstate high-cardinality
            continuous features. SHAP values provide model-agnostic attribution of each
            feature's contribution to individual predictions — more reliable for
            regulatory explainability (GDPR, SR 11-7).
        """)

    with tab_corr:
        section("PEARSON CORRELATION WITH CHURN TARGET")
        num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
                    "HasCrCard","IsActiveMember","EstimatedSalary"]
        corr = raw[num_cols+["Exited"]].corr()["Exited"].drop("Exited").sort_values(key=abs, ascending=False)
        colors = [C["red"] if v<0 else C["sky"] for v in corr.values]
        fig_c = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in corr.values],
            textposition="outside", textfont=dict(family="DM Mono",size=10)
        ))
        fig_c.add_vline(x=0, line_color=C["border"], line_width=1.5)
        chart_layout(fig_c, height=340,
                     title="Pearson Correlation — Features vs Churn (Exited)")
        fig_c.update_layout(
            xaxis=dict(title="Correlation Coefficient", range=[-0.45,0.45]),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_c, use_container_width=True)

        # Full correlation matrix
        section("FULL FEATURE CORRELATION MATRIX")
        corr_full = raw[num_cols+["Exited"]].corr().round(3)
        fig_cm = go.Figure(go.Heatmap(
            z=corr_full.values,
            x=corr_full.columns.tolist(),
            y=corr_full.index.tolist(),
            colorscale=[[0,"rgba(185,28,28,0.8)"],[0.5,"rgb(255,255,255)"],[1,"rgba(46,134,171,0.8)"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr_full.values.round(2),
            texttemplate="%{text}",
            textfont=dict(family="DM Mono",size=8),
            showscale=True,
            colorbar=dict(thickness=12, tickfont=dict(size=9))
        ))
        chart_layout(fig_cm, height=380, title="Feature Correlation Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab_eng:
        section("ENGINEERED FEATURES — DEFINITION & RATIONALE")
        eng_features = [
            ("BalanceToSalaryRatio", "Balance / EstimatedSalary",
             "Measures financial dependence on the bank. High ratio → customer has more at stake → but also high-churn in this dataset."),
            ("ProductDensity", "NumOfProducts / max(Tenure, 1)",
             "Products acquired per year of relationship. High density → rapid product bundling often precedes churn."),
            ("EngagementScore", "IsActiveMember + HasCrCard",
             "Composite engagement index 0–2. Low scores identify disengaged customers for re-activation."),
            ("AgeTenureInteraction", "Age / max(Tenure, 1)",
             "Older customers with short tenure — the highest-risk demographic combination."),
            ("ZeroBalance", "1 if Balance = 0, else 0",
             "Binary flag for dormant accounts — indicator of silent disengagement."),
            ("AgeGroup_Senior", "1 if Age ≥ 45, else 0",
             "Key age threshold where churn rate increases sharply. Reduces model reliance on raw age curve."),
            ("MultiProductRisk", "1 if NumOfProducts ≥ 3, else 0",
             "Flags the product overload cliff — 82.7%–100% churn territory."),
            ("InactiveHighBalance", "1 if Inactive AND Balance > median",
             "High-value disengaged customers — premium CLV at highest immediate risk."),
        ]
        html_rows = ""
        for name, formula, rationale in eng_features:
            html_rows += f"""
            <tr>
                <td style="font-family:'DM Mono',monospace;font-size:0.82rem;
                           color:{C['sky']};font-weight:500;">{name}</td>
                <td style="font-family:'DM Mono',monospace;font-size:0.8rem;
                           color:{C['slate']};">{formula}</td>
                <td style="font-size:0.83rem;color:{C['ink']};">{rationale}</td>
            </tr>"""
        st.markdown(f"""
        <table class="pro-table">
            <thead><tr><th>Feature Name</th><th>Formula</th><th>Business Rationale</th></tr></thead>
            <tbody>{html_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
def page_whatif(A):
    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">Scenario Analysis</div>
        <h1>Retention Intervention Simulator</h1>
        <p class="subtitle">
            Adjust customer attributes to model the impact of specific retention
            actions on churn probability before deploying resources.
        </p>
    </div>
    """, unsafe_allow_html=True)

    w1, w2 = st.columns([1, 1.1])

    with w1:
        section("BASELINE CUSTOMER PROFILE")
        wb1, wb2 = st.columns(2)
        with wb1:
            b_cr = st.slider("Credit Score",  300, 850, 600, 10, key="wi_cr")
            b_ag = st.slider("Age",            18,  92,  45,  1,  key="wi_ag")
            b_tn = st.slider("Tenure (yrs)",   0,   10,  3,   1,  key="wi_tn")
            b_np = st.selectbox("Products",  [1,2,3,4], index=0, key="wi_np")
        with wb2:
            b_bl = st.number_input("Balance (€)",  0.0, 300000.0, 120000.0, 5000.0, key="wi_bl")
            b_sl = st.number_input("Salary (€)",   0.0, 250000.0, 70000.0,  5000.0, key="wi_sl")
            b_cc = st.selectbox("Credit Card",   ["Yes","No"],       key="wi_cc")
            b_ac = st.selectbox("Active Member", ["No","Yes"],        key="wi_ac")
        b_ge = st.selectbox("Geography", ["France","Germany","Spain"], index=1, key="wi_ge")
        b_gn = st.selectbox("Gender",    ["Female","Male"],            key="wi_gn")

    base = dict(
        CreditScore=b_cr, Age=b_ag, Tenure=b_tn, Balance=b_bl,
        NumOfProducts=b_np, HasCrCard=1 if b_cc=="Yes" else 0,
        IsActiveMember=1 if b_ac=="Yes" else 0,
        EstimatedSalary=b_sl, Geography=b_ge, Gender=b_gn
    )
    bp = predict_single(A["best_model"], A["scaler"], A["feats"], base)

    with w2:
        section("CUSTOM INTERVENTION LEVER")
        lever = st.selectbox("Feature to adjust",
            ["IsActiveMember","NumOfProducts","Tenure",
             "CreditScore","HasCrCard","Balance"], key="wi_lv")

        if lever == "IsActiveMember":
            lv_v = st.radio("New value", [0,1],
                             format_func=lambda x:"Inactive(0)" if x==0 else "Active(1)",
                             horizontal=True, key="wi_lv_am")
        elif lever == "NumOfProducts":
            lv_v = st.slider("New number of products", 1, 4, 2, 1, key="wi_lv_np")
        elif lever == "Tenure":
            lv_v = st.slider("New tenure (years)", 0, 10, 7, 1, key="wi_lv_tn")
        elif lever == "CreditScore":
            lv_v = st.slider("New credit score", 300, 850, 720, 10, key="wi_lv_cr")
        elif lever == "HasCrCard":
            lv_v = st.radio("Has credit card", [0,1],
                             format_func=lambda x:"No" if x==0 else "Yes",
                             horizontal=True, key="wi_lv_cc")
        else:  # Balance
            lv_v = st.number_input("New balance (€)", 0.0, 300000.0,
                                    min(b_bl*0.5, 60000.0), 5000.0, key="wi_lv_bl")

    # Build all scenarios
    def sc(overrides):
        c = base.copy(); c.update(overrides)
        return predict_single(A["best_model"], A["scaler"], A["feats"], c)

    s_custom = base.copy(); s_custom[lever] = lv_v

    scenarios = [
        ("Baseline",                bp),
        ("Activate Membership",     sc({"IsActiveMember":1})),
        ("Add Credit Card",         sc({"HasCrCard":1})),
        ("Reduce to 2 Products",    sc({"NumOfProducts":2})),
        ("Increase Tenure +3 yrs",  sc({"Tenure":min(b_tn+3,10)})),
        ("Combined (Act+CC+2Prod)", sc({"IsActiveMember":1,"HasCrCard":1,"NumOfProducts":min(2,b_np)})),
        (f"Custom: {lever}={lv_v}", sc({lever: lv_v})),
    ]
    sc_names  = [s[0] for s in scenarios]
    sc_probs  = [s[1] for s in scenarios]
    sc_deltas = [round((p-bp)*100,1) for p in sc_probs]
    bar_colors= [C["slate"]] + [
        C["green"] if d<-2 else C["amber"] if d<0 else C["orange"] if d<5 else C["red"]
        for d in sc_deltas[1:]
    ]

    # Chart
    section("SCENARIO PROBABILITY COMPARISON")
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=sc_names, y=sc_probs, marker_color=bar_colors, marker_opacity=0.88,
        text=[f"{p:.1%}" for p in sc_probs],
        textposition="outside", textfont=dict(family="DM Mono",size=10),
        hovertemplate="%{x}<br>Probability: %{y:.2%}<extra></extra>"
    ))
    fig_sc.add_hline(
        y=bp, line_dash="dash", line_color=C["slate"], line_width=1.5,
        annotation_text=f"Baseline: {bp:.1%}",
        annotation_position="top right",
        annotation_font=dict(size=11, color=C["slate"])
    )
    chart_layout(fig_sc, height=360,
                 title="Retention Action Impact on Churn Probability")
    fig_sc.update_layout(
        yaxis=dict(tickformat=".0%", title="Churn Probability",
                   range=[0, min(1.0, max(sc_probs)+0.15)]),
        xaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Delta table
    section("SCENARIO SUMMARY TABLE")
    delta_df = pd.DataFrame({
        "Scenario":      sc_names,
        "Probability":   [f"{p:.2%}" for p in sc_probs],
        "Δ vs Baseline": [f"{d:+.1f}pp" for d in sc_deltas],
        "Risk Band":     [risk_band(p) for p in sc_probs],
        "Action":        ["—" if i==0 else
                          "✅ Improvement" if sc_deltas[i]<-1 else
                          "⚠️ Marginal"   if sc_deltas[i]<1 else
                          "🔴 Deterioration"
                          for i in range(len(sc_probs))]
    })
    st.dataframe(delta_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — CUSTOMER REGISTER
# ══════════════════════════════════════════════════════════════════════════════
def page_register(A):
    scored = A["scored"]

    st.markdown(f"""
    <div class="page-hero">
        <div class="badge">Customer Risk Register</div>
        <h1>Scored Customer Portfolio</h1>
        <p class="subtitle">
            All 10,000 customers scored with churn probability ·
            Filter by risk band, geography, and probability threshold
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    section("FILTERS")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        band_filter = st.multiselect(
            "Risk Band", ["Critical Risk","High Risk","Moderate Risk","Low Risk"],
            default=["Critical Risk","High Risk"], key="reg_band"
        )
    with f2:
        geo_filter = st.multiselect(
            "Geography", ["France","Germany","Spain"],
            default=["France","Germany","Spain"], key="reg_geo"
        )
    with f3:
        prob_min = st.slider("Min Probability", 0.0, 1.0, 0.40, 0.05, key="reg_pmin")
    with f4:
        top_n = st.selectbox("Show top N", [50,100,200,500,"All"], index=1, key="reg_n")

    # Apply filters
    mask = (
        scored["RiskBand"].isin(band_filter) &
        scored["Geography"].isin(geo_filter) &
        (scored["ChurnProbability"] >= prob_min)
    )
    filtered = scored[mask].sort_values("ChurnProbability", ascending=False)
    if top_n != "All":
        filtered = filtered.head(int(top_n))

    # Summary strip
    st.markdown("<br>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    with s1: kpi("Customers Shown",   f"{len(filtered):,}", "After filters", "blue")
    with s2: kpi("Avg Probability",   f"{filtered['ChurnProbability'].mean():.1%}",
                  "Filtered group", "amber")
    with s3: kpi("Actual Churners",   f"{filtered['Exited'].sum():,}",
                  "Known churned", "red")
    with s4: kpi("Capture Rate",
                  f"{filtered['Exited'].sum()/A['summ']['actual_churners']:.1%}",
                  "Of total churners", "gold")

    st.markdown("<br>", unsafe_allow_html=True)

    # Display table
    section(f"RISK REGISTER — {len(filtered):,} CUSTOMERS")
    display_df = filtered[[
        "CustomerId","Surname","Geography","Gender","Age",
        "CreditScore","Balance","NumOfProducts","IsActiveMember",
        "ChurnProbability","RiskBand","Exited"
    ]].copy()
    display_df["Balance"]          = display_df["Balance"].apply(lambda x: f"€{x:,.0f}")
    display_df["ChurnProbability"] = display_df["ChurnProbability"].apply(lambda x: f"{x:.1%}")
    display_df["IsActiveMember"]   = display_df["IsActiveMember"].map({1:"✓ Active",0:"✗ Inactive"})
    display_df["Exited"]           = display_df["Exited"].map({1:"✓ Churned",0:"Retained"})
    display_df.columns = [
        "Customer ID","Surname","Country","Gender","Age",
        "Credit Score","Balance","Products","Member Status",
        "Churn Prob.","Risk Band","Actual Status"
    ]

    st.dataframe(display_df, use_container_width=True,
                 height=500, hide_index=True)

    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇  Download Filtered Register (CSV)",
        data=csv,
        file_name="churn_risk_register_filtered.csv",
        mime="text/csv"
    )

    # Distribution of filtered group
    section("FILTERED GROUP — PROBABILITY DISTRIBUTION")
    fig_dist = go.Figure()
    for rb, col in [("Critical Risk",C["red"]),("High Risk",C["orange"]),
                    ("Moderate Risk",C["amber"]),("Low Risk",C["green"])]:
        sub = filtered[filtered["RiskBand"]==rb]
        if len(sub) > 0:
            fig_dist.add_trace(go.Histogram(
                x=sub["ChurnProbability"], name=rb,
                marker_color=col, opacity=0.72, nbinsx=30
            ))
    chart_layout(fig_dist, height=280,
                 title="Probability Distribution — Filtered Portfolio")
    fig_dist.update_layout(
        barmode="stack",
        xaxis=dict(title="Churn Probability"),
        yaxis=dict(title="Count"),
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ─── ROUTER ──────────────────────────────────────────────────────────────────
def main():
    A    = load_all()
    page = sidebar()

    if   "Executive Dashboard" in page: page_dashboard(A)
    elif "EDA"                 in page: page_eda(A)
    elif "Risk Calculator"     in page: page_calculator(A)
    elif "Model Performance"   in page: page_models(A)
    elif "Feature Intelligence"in page: page_features(A)
    elif "What-If"             in page: page_whatif(A)
    elif "Customer Register"   in page: page_register(A)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<small style='color:{C['slate']};font-family:DM Sans;'>"
        f"Churn Intelligence System &nbsp;·&nbsp; "
        f"Gradient Boosting · SMOTE · Isotonic Calibration &nbsp;·&nbsp; "
        f"ROC-AUC: {summary['best_roc_auc']:.4f} &nbsp;·&nbsp; "
        f"10,000 European bank customers &nbsp;·&nbsp; "
        f"Created by Anoop Puri"
        f"</small>", unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
