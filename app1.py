"""Loan Eligibility Predictor - Streamlit UI (dark theme, coral accent).

Files needed in the same folder: app.py, styles.css, rf_model.pkl, Copy_of_loan.xlsx
"""
import difflib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
except Exception:      # xgboost not installed: that model is skipped
    XGBClassifier = None

st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------
# PALETTE (must match styles.css)
# ------------------------------------------------------------------
BG, CARD, INPUT, BORDER = "#1A1A1C", "#232326", "#2B2B2F", "#333337"
TEXT, MUTED = "#EDEDEE", "#8E8E96"
ACCENT, ACCENT_HOVER = "#DF5148", "#E8665D"      # coral accent (name kept for brevity)
APPROVED, REJECTED = ACCENT, "#7C93AD"           # approved = coral, not approved = slate blue
GRID = "rgba(255,255,255,0.06)"                 # very faint gridlines
NEUTRAL_BAR = "#5A5A60"                          # neutral grey for non-highlighted bars
WARN = "#E0B25A"

# ------------------------------------------------------------------
# CSS (UTF-8, injected once)
# ------------------------------------------------------------------
CSS_PATH = Path(__file__).parent / "styles1.css"
st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SHARED PLOTLY TEMPLATE
# ------------------------------------------------------------------
def plotly_template():
    """One template for every chart: transparent bg, Inter font, light gridlines."""
    axis = dict(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, automargin=True,
                tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED)))
    return go.layout.Template(layout=go.Layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        colorway=[ACCENT, REJECTED, WARN, NEUTRAL_BAR],
        xaxis=axis, yaxis=axis,
        legend=dict(bgcolor=CARD, font=dict(color=TEXT)),
        margin=dict(t=10, b=40, l=10, r=20),
        hoverlabel=dict(bgcolor=CARD, font=dict(color=TEXT)),
    ))

pio.templates["loan"] = plotly_template()
pio.templates.default = "loan"

# ------------------------------------------------------------------
# MODEL + DATA
# ------------------------------------------------------------------
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

FEATURE_NAMES = [
    "Gender", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
    "Property_Area_Semiurban", "Property_Area_Urban", "TotalIncome",
]
FRIENDLY = {
    "Gender": "Gender", "Dependents": "Dependents", "Education": "Education",
    "Self_Employed": "Self employed", "ApplicantIncome": "Applicant income",
    "CoapplicantIncome": "Coapplicant income", "LoanAmount": "Loan amount",
    "Loan_Amount_Term": "Loan term", "Credit_History": "Credit history",
    "Property_Area_Semiurban": "Semiurban property", "Property_Area_Urban": "Urban property",
    "TotalIncome": "Total income",
}
DATA_FILE = "Copy_of_loan.xlsx"


@st.cache_data
def load_raw_data():
    return pd.read_excel(DATA_FILE)


def get_data():
    try:
        return load_raw_data()
    except FileNotFoundError:
        st.error(f"Couldn't find `{DATA_FILE}`. Place it in the same folder as app.py.")
        st.stop()


@st.cache_data
def prepare_training(raw):
    """Same feature engineering as the notebook: fill gaps, encode, add TotalIncome."""
    d = raw.copy()
    d["Dependents"] = pd.to_numeric(d["Dependents"].astype(str).str.replace("+", "", regex=False),
                                    errors="coerce")
    for c in ["Gender", "Self_Employed", "Dependents", "Credit_History", "Education", "Property_Area"]:
        d[c] = d[c].fillna(d[c].mode().iloc[0])
    d["LoanAmount"] = d["LoanAmount"].fillna(d["LoanAmount"].median())
    d["Loan_Amount_Term"] = d["Loan_Amount_Term"].fillna(d["Loan_Amount_Term"].median())
    d["Gender"] = (d["Gender"] == "Male").astype(int)
    d["Education"] = (d["Education"] == "Graduate").astype(int)
    d["Self_Employed"] = (d["Self_Employed"] == "Yes").astype(int)
    d["Property_Area_Semiurban"] = (d["Property_Area"] == "Semiurban").astype(int)
    d["Property_Area_Urban"] = (d["Property_Area"] == "Urban").astype(int)
    d["TotalIncome"] = d["ApplicantIncome"] + d["CoapplicantIncome"]
    return d[FEATURE_NAMES], (d["Loan_Status"] == "Y").astype(int)


@st.cache_data
def evaluate_models():
    """Train 3 models on an 80/20 stratified split and score them on the hold-out 20%."""
    X, y = prepare_training(get_data())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Same 3 models and same split as the notebook (test_size=0.2, random_state=42, stratify)
    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    rows, extra = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
            "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred),
            "ROC-AUC": roc_auc_score(yte, proba),
        })
        if name == "Random Forest":
            fpr, tpr, _ = roc_curve(yte, proba)
            extra = {"cm": confusion_matrix(yte, pred, labels=[0, 1]), "fpr": fpr, "tpr": tpr,
                     "auc": roc_auc_score(yte, proba)}
    return rows, extra


# ------------------------------------------------------------------
# REUSABLE UI HELPERS
# ------------------------------------------------------------------
def page_header(title, description):
    st.markdown(f'<div class="page-head"><div class="page-title">{title}</div>'
                f'<div class="page-desc">{description}</div></div>', unsafe_allow_html=True)


def section_header(title, subtitle=None):
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="sec-title">{title}</div>{sub}', unsafe_allow_html=True)


def kpi_card(label, value, caption=None, accent=False):
    cap = f'<div class="kpi-cap">{caption}</div>' if caption else ""
    cls = "kpi-value accent" if accent else "kpi-value"
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="{cls}">{value}</div>{cap}</div>', unsafe_allow_html=True)


def insight_chart(fig, title, caption=None, height=320):
    """Card with an insight-style title, one-line caption and the chart."""
    with st.container(border=True):
        cap = f'<div class="chart-cap">{caption}</div>' if caption else ""
        st.markdown(f'<div class="chart-title">{title}</div>{cap}', unsafe_allow_html=True)
        fig.update_layout(height=height, template=pio.templates["loan"], paper_bgcolor=CARD, plot_bgcolor=CARD,
                          font=dict(family="Inter, sans-serif", color=TEXT, size=12))
        fig.update_xaxes(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                         tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
        fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                         tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def html_table(headers, rows, best_idx=None, num_cols=()):
    """Clean HTML table. num_cols = column indexes aligned right."""
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{"num" if i in num_cols else ""}">{v}</td>' for i, v in enumerate(row))
        body += f'<tr class="{"best" if r == best_idx else ""}">{tds}</tr>'
    return f'<table class="clean-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def steps_line(steps):
    parts = []
    for i, s in enumerate(steps, 1):
        parts.append(f'<div class="step"><div class="step-num">{i}</div><div class="step-text">{s}</div></div>')
        if i < len(steps):
            parts.append('<span class="step-arrow">&rarr;</span>')
    st.markdown(f'<div class="steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def go_to(page):
    """Button callback: change the active page."""
    st.session_state.nav_page = page


# ------------------------------------------------------------------
# SESSION STATE + SIDEBAR NAV
# ------------------------------------------------------------------
PAGES = ["Overview", "Data Overview", "Data Insights", "Loan Form",
         "Bulk Prediction"]
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.markdown('<div class="brand">Loan Predictor</div>'
                '<div class="brand-sub">Predict loan eligibility using machine learning.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-label">Navigate</div>', unsafe_allow_html=True)
    st.radio("Navigate", PAGES, key="nav_page", label_visibility="collapsed")
    st.markdown('<div class="side-label">Model</div><div class="side-note">Random Forest (best accuracy of the models tested)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-label">Note</div><div class="side-note">'
                'Model predictions are decision support, not a final credit decision.</div>',
                unsafe_allow_html=True)

page = st.session_state.nav_page


# ==================================================================
# PAGE: OVERVIEW
# ==================================================================
if page == "Overview":
    page_header("Overview", f"Loan eligibility predictor &middot; {datetime.now().strftime('%A, %d %b %Y')}")

    raw = get_data()
    _, y_all = prepare_training(raw)
    approval_rate = y_all.mean() * 100
    baseline = max(y_all.mean(), 1 - y_all.mean()) * 100
    rows, _ = evaluate_models()
    rf = next(r for r in rows if r["Model"] == "Random Forest")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Model accuracy", f"{rf['Accuracy']*100:.1f}%", "20% hold-out test set")
    with c2: kpi_card("Baseline accuracy", f"{baseline:.1f}%", "Always predicting the majority class")
    with c3: kpi_card("Dataset size", f"{len(raw):,}", "Loan applications")
    with c4: kpi_card("Approval rate in data", f"{approval_rate:.1f}%", "Share of approved loans")

    section_header("Model quality", "Random Forest, scored on 20% of the data the model did not train on.")
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi_card("Precision", f"{rf['Precision']*100:.1f}%")
    with m2: kpi_card("Recall", f"{rf['Recall']*100:.1f}%")
    with m3: kpi_card("F1 score", f"{rf['F1']*100:.1f}%")
    

    with st.container(border=True):
        section_header("How it works")
        steps_line(["Enter details", "Preprocess", "Model predicts", "Result with confidence"])
        st.button("Check eligibility", type="primary", key="ov_cta", on_click=go_to, args=("Loan Form",))

    with st.container(border=True):
        section_header("Your recent predictions", "Predictions made in this session.")
        if not st.session_state.history:
            st.markdown('<div class="empty">No predictions yet. Use the Loan Form to make one.</div>',
                        unsafe_allow_html=True)
        else:
            recent = st.session_state.history[-5:][::-1]
            trs = [[h["time"], h["result"], f"{h['probability']:.0f}%", f"₹{h['loan_amount']:,.0f}k"] for h in recent]
            st.markdown(html_table(["Time", "Result", "Approval probability", "Loan amount"], trs,
                                   num_cols=(2, 3)), unsafe_allow_html=True)


# ==================================================================
# PAGE: DATA OVERVIEW
# ==================================================================
elif page == "Data Overview":
    raw = get_data()
    total_rows, total_cols = len(raw), len(raw.columns)
    missing_by_col = raw.isnull().sum()
    total_missing = int(missing_by_col.sum())
    completeness = (1 - total_missing / (total_rows * total_cols)) * 100
    dup = int(raw.duplicated().sum())
    n_app = int((raw["Loan_Status"] == "Y").sum())
    n_rej = int((raw["Loan_Status"] == "N").sum())

    page_header("Data Overview", "Loan prediction dataset: columns, types and data health.")

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Rows", f"{total_rows}", "Applications")
    with k2: kpi_card("Completeness", f"{completeness:.1f}%", f"{total_missing:,} empty cells")
    with k3: kpi_card("Duplicates", f"{dup}", "None found" if dup == 0 else "Rows repeated")
    with k4: kpi_card("Class balance", f"{n_app / max(n_rej, 1):.1f} : 1", "Approved to rejected")

    # ---- column classification ----
    CATEGORICAL_LIKE = {"Credit_History"}      # stored as 0/1 but is a category
    meanings = {
        "Loan_ID": "Unique application number", "Gender": "Male or Female", "Married": "Marital status",
        "Dependents": "0, 1, 2 or 3+", "Education": "Graduate or Not Graduate", "Self_Employed": "Yes or No",
        "ApplicantIncome": "Applicant monthly income", "CoapplicantIncome": "Coapplicant monthly income",
        "LoanAmount": "Requested amount (in thousands)", "Loan_Amount_Term": "Repayment term in months",
        "Credit_History": "1 = past credit bills paid on time, 0 = not", "Property_Area": "Urban, Semiurban or Rural",
        "Loan_Status": "Approved (Y) or not (N)",
    }

    def col_type(c):
        if c == "Loan_ID": return "ID"
        if c == "Loan_Status": return "Target"
        if c in CATEGORICAL_LIKE or not pd.api.types.is_numeric_dtype(raw[c]): return "Categorical"
        return "Numeric"

    types = {c: col_type(c) for c in raw.columns}
    counts = pd.Series(types).value_counts()
    type_line = (f"{counts.get('Categorical', 0)} categorical, {counts.get('Numeric', 0)} numeric, "
                 f"{counts.get('ID', 0)} ID, {counts.get('Target', 0)} target")

    left, right = st.columns([1.5, 1])
    with left:
        with st.container(border=True):
            section_header("Data dictionary", f"{total_cols} columns: {type_line}.")
            drows = [[c, f'<span style="color:{MUTED}">{types[c]}</span>', meanings.get(c, "Dataset variable"),
                      int(raw[c].isnull().sum())] for c in raw.columns]
            st.markdown(html_table(["Column", "Type", "Meaning", "Missing"], drows, num_cols=(3,)),
                        unsafe_allow_html=True)

    with right:
        miss = missing_by_col[missing_by_col > 0].sort_values(ascending=True)   # largest ends on top
        if len(miss):
            fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h",
                                   marker=dict(color=ACCENT), text=miss.values,
                                   textposition="outside", cliponaxis=False))
            fig.update_layout(showlegend=False, margin=dict(t=10, b=40, l=10, r=40),
                              xaxis=dict(title="Missing values", range=[0, miss.max() * 1.2]),
                              yaxis=dict(title=""))
            insight_chart(fig, f"{miss.index[-1]} has the most gaps ({int(miss.iloc[-1])} values)",
                          "Gaps are filled with the median (numbers) or most common value (categories).",
                          height=330)
        else:
            st.success("No missing values found.")

    b1, b2 = st.columns([1.25, 1])
    with b1:
        with st.container(border=True):
            section_header("Numeric summary", "Income and loan amount statistics.")
            num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
            names = ["Applicant income", "Coapplicant income", "Loan amount", "Term (months)"]
            stats = [("Mean", lambda s: s.mean()), ("Median", lambda s: s.median()),
                     ("Std dev", lambda s: s.std()), ("Min", lambda s: s.min()), ("Max", lambda s: s.max())]
            srows = [[lab] + [f"{fn(raw[c]):,.0f}" for c in num_cols] for lab, fn in stats]
            st.markdown(html_table([""] + names, srows, num_cols=(1, 2, 3, 4)), unsafe_allow_html=True)
    with b2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[n_app], y=["Approved"], orientation="h", marker=dict(color=APPROVED),
                             text=[n_app], textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(x=[n_rej], y=["Not approved"], orientation="h", marker=dict(color=REJECTED),
                             text=[n_rej], textposition="outside", cliponaxis=False))
        fig.update_layout(showlegend=False, margin=dict(t=10, b=40, l=10, r=40),
                          xaxis=dict(title="Applications", range=[0, max(n_app, n_rej) * 1.2]),
                          yaxis=dict(title="", autorange="reversed"))
        insight_chart(fig, f"About {n_app / total_rows * 100:.0f}% of applications were approved",
                      f"Loan_Status across all {total_rows} applications.", height=260)

    with st.container(border=True):
        section_header("Data preparation pipeline")
        steps_line(["Check shape, types, nulls", "Fill gaps (median / mode)", "Encode categories",
                    "Add TotalIncome", "Split train / test"])


# ==================================================================
# PAGE: DATA INSIGHTS
# ==================================================================
elif page == "Data Insights":
    raw = get_data()
    total = len(raw)
    n_app = int((raw["Loan_Status"] == "Y").sum())
    approval = n_app / total * 100
    page_header("Data Insights", "Patterns in the training data and what drives approval.")

    k1, k2, k3 = st.columns(3)
    with k1: kpi_card("Total applications", f"{total}", "Rows in the training data")
    with k2: kpi_card("Approval rate", f"{approval:.1f}%", f"{n_app} approved loans", accent=True)
    with k3: kpi_card("Avg loan amount", f"₹{raw['LoanAmount'].mean():,.0f}k", "Average requested")

    def rate_fig(col, order):
        """Approval rate (%) per category, one colour, dashed line = overall average."""
        d = raw.dropna(subset=[col])
        g = d.groupby(d[col].astype(str))["Loan_Status"].agg(rate=lambda x: (x == "Y").mean() * 100, n="count")
        order = [o for o in order if o in g.index]
        fig = go.Figure(go.Bar(
            x=[f"{o}<br>(n={int(g.loc[o, 'n'])})" for o in order], y=[g.loc[o, "rate"] for o in order],
            marker=dict(color=ACCENT), width=0.5, text=[f"{g.loc[o, 'rate']:.1f}%" for o in order],
            textposition="outside", cliponaxis=False))
        fig.add_hline(y=approval, line_dash="dash", line_color=MUTED)
        fig.update_layout(showlegend=False, yaxis=dict(title="Approval rate (%)", range=[0, 112]),
                          margin=dict(t=20, b=50, l=55, r=10))
        return fig

    # ---- Credit history: rate + counts ----
    ch = raw.dropna(subset=["Credit_History"])
    grp = ch.groupby("Credit_History")["Loan_Status"].agg(rate=lambda x: (x == "Y").mean() * 100, n="count")
    r0, r1 = grp.loc[0.0, "rate"], grp.loc[1.0, "rate"]
    n_unknown = int(raw["Credit_History"].isna().sum())
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=[f"No credit history<br>(n={int(grp.loc[0.0, 'n'])})", f"Has credit history<br>(n={int(grp.loc[1.0, 'n'])})"],
            y=[r0, r1], marker=dict(color=[NEUTRAL_BAR, ACCENT]), width=0.45,
            text=[f"{r0:.0f}%", f"{r1:.0f}%"], textposition="outside", cliponaxis=False))
        fig.update_layout(showlegend=False, yaxis=dict(title="Approval rate (%)", range=[0, 112]),
                          margin=dict(t=20, b=50, l=55, r=10))
        insight_chart(fig, f"With credit history {r1:.0f}% are approved, without only {r0:.0f}%",
                      f"Strongest signal in the data. {n_unknown} applicants with unknown history are excluded.", height=340)
    with c2:
        ct = pd.crosstab(ch["Credit_History"], ch["Loan_Status"]).reindex([0.0, 1.0], fill_value=0)
        ct = ct.reindex(columns=["Y", "N"], fill_value=0)
        xs = ["No credit history", "Has credit history"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=xs, y=ct["Y"], name="Approved", marker_color=APPROVED, text=ct["Y"],
                             textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(x=xs, y=ct["N"], name="Not approved", marker_color=REJECTED, text=ct["N"],
                             textposition="outside", cliponaxis=False))
        fig.update_layout(barmode="group", yaxis=dict(title="Applicants"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                          margin=dict(t=30, b=50, l=55, r=10))
        insight_chart(fig, "Credit History vs Loan Status", "Number of applicants in each group.", height=340)

    # ---- one grouped chart for all categories ----
    cats = [("Property", "Property_Area"), ("Education", "Education"),
            ("Self employed", "Self_Employed"), ("Married", "Married"), ("Dependents", "Dependents")]
    items = []
    for label, col in cats:
        d = raw.dropna(subset=[col])
        for val, g in d.groupby(d[col].astype(str)):
            items.append((f"{label}: {val}  (n={len(g)})", (g["Loan_Status"] == "Y").mean() * 100))
    items.sort(key=lambda t: t[1])
    colors = [ACCENT if abs(r - approval) >= 5 else NEUTRAL_BAR for _, r in items]
    fig = go.Figure(go.Bar(x=[r for _, r in items], y=[n for n, _ in items], orientation="h",
                           marker=dict(color=colors), text=[f"{r:.1f}%" for _, r in items],
                           textposition="outside", cliponaxis=False))
    fig.add_vline(x=approval, line_dash="dash", line_color=MUTED,
                  annotation_text=f"Average {approval:.1f}%", annotation_position="top",
                  annotation_font_color=MUTED)
    fig.update_layout(showlegend=False, xaxis=dict(title="Approval rate (%)", range=[0, 105]),
                      yaxis=dict(title=""), margin=dict(t=30, b=40, l=10, r=40))
    insight_chart(fig, "Approval rate by applicant group",
                  "Coral bars differ from the average by 5 points or more. Grey bars are close to average.",
                  height=130 + 28 * len(items))

    # ---- single-variable approval rate charts (from your earlier dashboard) ----
    d1, d2, d3 = st.columns(3)
    with d1:
        insight_chart(rate_fig("Property_Area", ["Rural", "Semiurban", "Urban"]),
                      "Approval Rate by Property Area", "Dashed line is the overall average.", height=320)
    with d2:
        insight_chart(rate_fig("Education", ["Graduate", "Not Graduate"]),
                      "Approval Rate by Education", "Dashed line is the overall average.", height=320)
    with d3:
        insight_chart(rate_fig("Self_Employed", ["No", "Yes"]),
                      "Approval Rate by Self Employed", "Dashed line is the overall average.", height=320)

    # ---- scatter ----
    sc = go.Figure()
    for status, name, color in [("Y", "Approved", APPROVED), ("N", "Not approved", REJECTED)]:
        d = raw[raw["Loan_Status"] == status]
        sc.add_trace(go.Scatter(x=d["ApplicantIncome"], y=d["LoanAmount"], mode="markers", name=name,
                                marker=dict(color=color, size=6, opacity=0.6),
                                hovertemplate="Income: ₹%{x:,.0f}<br>Loan: ₹%{y:,.0f}k<extra>" + name + "</extra>"))
    sc.update_layout(xaxis=dict(title="Applicant income (₹, log scale)", type="log"),
                     yaxis=dict(title="Loan amount (₹ thousands)"),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                     margin=dict(t=30, b=50, l=60, r=20))
    insight_chart(sc, "Applicant Income vs Loan Amount",
                  "Bigger incomes tend to request bigger loans, but approval is mixed at every level. Log scale on income.",
                  height=380)

    # ---- correlation heatmap + feature importance ----
    e1, e2 = st.columns(2)
    with e1:
        num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
        corr = raw[num_cols].corr()
        short = ["Applicant inc.", "Coapplicant inc.", "Loan amount", "Loan term", "Credit history"]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=short, y=short, zmin=-1, zmax=1, showscale=False,
            colorscale=[[0, REJECTED], [0.5, INPUT], [1, ACCENT]],
            text=[[f"{v:.2f}" for v in row] for row in corr.values], texttemplate="%{text}",
            textfont=dict(color=TEXT), hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10, b=60, l=10, r=10))
        insight_chart(fig, "Correlation Between Numeric Features",
                      "Loan amount moves with applicant income (0.57). Others are weakly related.", height=400)
    with e2:
        imp = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False).head(10)
        imp = imp.sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=imp.values * 100, y=[FRIENDLY[i] for i in imp.index], orientation="h",
                               marker=dict(color=ACCENT), text=[f"{v*100:.1f}%" for v in imp.values],
                               textposition="outside", cliponaxis=False))
        fig.update_layout(showlegend=False, xaxis=dict(title="Importance (%)", range=[0, imp.max() * 100 * 1.25]),
                          yaxis=dict(title=""), margin=dict(t=10, b=40, l=10, r=40))
        insight_chart(fig, f"{FRIENDLY[imp.index[-1]]} matters most to the Random Forest",
                      "Top 10 features by importance in the trained model.", height=400)


# ==================================================================
# PAGE: LOAN FORM
# ==================================================================
elif page == "Loan Form":
    page_header("Loan Application Form", "Individual applicant assessment: enter the details to estimate eligibility.")

    # Typical applicant from training data, used to explain which factors moved the decision
    X_train, _ = prepare_training(get_data())
    REF = X_train.median()
    GROUPS = {
        "Credit history": ["Credit_History"],
        "Income": ["ApplicantIncome", "CoapplicantIncome", "TotalIncome"],
        "Loan amount": ["LoanAmount"],
        "Loan term": ["Loan_Amount_Term"],
        "Property area": ["Property_Area_Semiurban", "Property_Area_Urban"],
        "Education": ["Education"], "Dependents": ["Dependents"],
        "Employment type": ["Self_Employed"], "Gender": ["Gender"],
    }

    def explain(row, base_p):
        out = []
        for name, cols in GROUPS.items():
            alt = row.copy()
            for c in cols:
                alt[c] = REF[c]
            delta = base_p - rf_model.predict_proba(alt)[0][1]
            if abs(delta) >= 0.01:
                out.append((name, delta))
        return sorted(out, key=lambda t: abs(t[1]), reverse=True)[:3]

    with st.form("loan_form"):
        with st.container(border=True):
            section_header("Personal information")
            a, b = st.columns(2)
            gender = a.selectbox("Gender", ["Female", "Male"], index=None, placeholder="Select gender")
            dependents = b.selectbox("Dependents", [0, 1, 2, 3], index=None, placeholder="Select number of dependents",
                                     help="People financially dependent on the applicant (3 means 3 or more).")
            a, b = st.columns(2)
            education = a.selectbox("Education", ["Not Graduate", "Graduate"], index=None,
                                    placeholder="Select education level")
            self_employed = b.selectbox("Self employed", ["No", "Yes"], index=None,
                                        placeholder="Select employment status")

        with st.container(border=True):
            section_header("Financial information")
            a, b = st.columns(2)
            applicant_income = a.number_input("Applicant income (₹ per month)", min_value=0.0, value=None,
                                              step=1000.0, format="%.0f", placeholder="e.g. 5000")
            coapplicant_income = b.number_input("Coapplicant income (₹ per month)", min_value=0.0, value=0.0,
                                                step=1000.0, format="%.0f", help="Leave 0 if there is no coapplicant.")
            a, b = st.columns(2)
            loan_amount = a.number_input("Loan amount (₹ in thousands)", min_value=0.0, value=None, step=10.0,
                                         format="%.0f", placeholder="e.g. 150",
                                         help="Enter in thousands. 150 means ₹1,50,000. The model was trained in this unit.")
            loan_term = b.selectbox("Loan term", [120, 180, 240, 300, 360], index=4,
                                    format_func=lambda m: f"{m} months ({m // 12} years)")

        with st.container(border=True):
            section_header("Credit and property information")
            a, b = st.columns(2)
            credit_history = a.selectbox("Previous loan/credit bills paid on time?", ["Yes", "No"], index=None,
                                         placeholder="Select credit history")
            property_area = b.selectbox("Property area", ["Rural", "Semiurban", "Urban"], index=None,
                                        placeholder="Select property area")

        submit = st.form_submit_button("Check loan eligibility", type="primary", width="stretch")

    if submit:
        errors = []
        if None in (gender, dependents, education, self_employed, credit_history, property_area):
            errors.append("Please fill in every dropdown.")
        if not applicant_income or applicant_income <= 0:
            errors.append("Applicant income must be greater than 0.")
        if not loan_amount or loan_amount <= 0:
            errors.append("Loan amount must be greater than 0.")
        for e in errors:
            st.error(e)

        if not errors:
            semi = 1 if property_area == "Semiurban" else 0
            urban = 1 if property_area == "Urban" else 0
            total_income = applicant_income + coapplicant_income
            row = pd.DataFrame([[
                1 if gender == "Male" else 0, dependents, 1 if education == "Graduate" else 0,
                1 if self_employed == "Yes" else 0, applicant_income, coapplicant_income, loan_amount,
                loan_term, 1.0 if credit_history == "Yes" else 0.0, semi, urban, total_income,
            ]], columns=FEATURE_NAMES)
            proba = rf_model.predict_proba(row)[0]            # [P(not approved), P(approved)]
            p_app = float(proba[1]) * 100
            ok = rf_model.predict(row)[0] == 1
            label = "Likely approved" if ok else "Likely not approved"
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"), "result": label,
                "probability": p_app, "loan_amount": loan_amount})

            left, right = st.columns(2)
            with left:
                with st.container(border=True):
                    text = ("The model predicts that this application is eligible." if ok
                            else "The model predicts that this application is not eligible.")
                    st.markdown(
                        f'<div class="result-title {"ok" if ok else "no"}">{label}</div>'
                        f'<div class="sec-sub">{text}</div>'
                        f'<div class="progress-label" style="display:flex;justify-content:space-between;font-size:13px;margin-top:14px">'
                        f'<span>Approval probability</span><span>{p_app:.1f}%</span></div>'
                        f'<div class="bar-track"><div class="bar-fill" style="width:{p_app:.0f}%;'
                        f'background:{APPROVED if ok else REJECTED}"></div></div>', unsafe_allow_html=True)
                    factors = explain(row, p_app / 100)
                    if factors:
                        st.markdown('<div class="sec-sub" style="margin-top:16px">Top factors (compared with a typical applicant)</div>',
                                    unsafe_allow_html=True)
                        html = ""
                        for name, d in factors:
                            word = '<span class="up">raises</span>' if d > 0 else '<span class="down">lowers</span>'
                            html += f'<div class="factor">{name} {word} approval chance by about {abs(d)*100:.0f} points</div>'
                        st.markdown(html, unsafe_allow_html=True)
            with right:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[proba[1] * 100], y=["Approved"], orientation="h", marker=dict(color=APPROVED),
                                     text=[f"{proba[1]*100:.1f}%"], textposition="outside", cliponaxis=False))
                fig.add_trace(go.Bar(x=[proba[0] * 100], y=["Not approved"], orientation="h", marker=dict(color=REJECTED),
                                     text=[f"{proba[0]*100:.1f}%"], textposition="outside", cliponaxis=False))
                fig.update_layout(showlegend=False, xaxis=dict(range=[0, 115], title="Probability (%)"),
                                  yaxis=dict(title="", autorange="reversed"), margin=dict(t=10, b=40, l=10, r=40))
                insight_chart(fig, "Prediction confidence",
                              "Model estimate, not a final credit decision.", height=250)


# ==================================================================
# PAGE: BULK PREDICTION
# ==================================================================
elif page == "Bulk Prediction":
    page_header("Bulk Prediction", "Upload a CSV, Excel or JSON file to predict many applicants at once.")

    # ---------------- CONFIG ----------------
    REQUIRED_RAW_COLUMNS = [
        "Gender", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area",
    ]
    CRITICAL_COLUMNS = ["Credit_History", "ApplicantIncome", "LoanAmount"]
    NONCRITICAL_COLUMNS = [c for c in REQUIRED_RAW_COLUMNS if c not in CRITICAL_COLUMNS]
    CATEGORICAL_COLUMNS = ["Gender", "Education", "Self_Employed", "Property_Area"]
    NUMERIC_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    NOT_IN_FILE = "-- Not in my file --"
    NO_LABEL = "-- None --"

    COLUMN_ALIASES = {
        "Gender": ["gender", "sex"],
        "Dependents": ["dependents", "dependent", "no_of_dependents", "number_of_dependents",
                       "num_dependents", "family_dependents"],
        "Education": ["education", "education_level", "edu", "degree", "qualification"],
        "Self_Employed": ["self_employed", "selfemployed", "employment", "employment_status", "job_status"],
        "ApplicantIncome": ["applicantincome", "applicant_income", "income", "salary",
                            "monthly_income", "annual_income"],
        "CoapplicantIncome": ["coapplicantincome", "coapplicant_income", "co_income",
                              "partner_income", "spouse_income"],
        "LoanAmount": ["loanamount", "loan_amount", "loan_amt", "requested_loan", "requested_amount"],
        "Loan_Amount_Term": ["loan_amount_term", "loan_term", "term", "loan_term_months",
                             "repayment_months", "tenure"],
        "Credit_History": ["credit_history", "credit_record", "credit_score", "cibil", "cibil_score"],
        "Property_Area": ["property_area", "property", "area", "location", "property_location"],
    }
    BASE_MAPS = {
        "Gender": {"male": "Male", "m": "Male", "female": "Female", "f": "Female"},
        "Education": {"graduate": "Graduate", "not graduate": "Not Graduate",
                      "not_graduate": "Not Graduate", "undergraduate": "Not Graduate"},
        "Self_Employed": {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "1.0": "Yes",
                          "self employed": "Yes", "self-employed": "Yes", "self_employed": "Yes",
                          "selfemployed": "Yes", "no": "No", "n": "No", "false": "No", "0": "No",
                          "0.0": "No", "employed": "No", "salaried": "No", "not self employed": "No",
                          "not self-employed": "No", "unemployed": "No"},
        "Property_Area": {"urban": "Urban", "semiurban": "Semiurban", "semi urban": "Semiurban",
                          "semi-urban": "Semiurban", "rural": "Rural"},
    }
    TARGET_OPTIONS = {
        "Gender": ["Male", "Female"], "Education": ["Graduate", "Not Graduate"],
        "Self_Employed": ["No", "Yes"], "Property_Area": ["Rural", "Semiurban", "Urban"],
    }
    FALLBACK_DEFAULTS = {
        "Gender": "Male", "Dependents": 0, "Education": "Graduate", "Self_Employed": "No",
        "CoapplicantIncome": 1200.0, "Loan_Amount_Term": 360.0, "Property_Area": "Semiurban",
    }
    SCALE_OPTIONS = {
        "No change": 1.0,
        "÷ 12  (annual → monthly)": 1 / 12,
        "÷ 1000  (rupees → thousands)": 1 / 1000,
        "× 12  (years → months)": 12.0,
        "× 1000  (thousands → rupees)": 1000.0,
    }
    POSITIVE_WORDS = {"y", "yes", "1", "1.0", "true", "t", "approved", "approve"}
    NEGATIVE_WORDS = {"n", "no", "0", "0.0", "false", "f", "rejected", "not approved", "denied", "declined"}
    LABEL_HINTS = {"loanstatus", "loanapproved", "approved", "status", "target", "label", "outcome",
                   "approval", "actual"}

    # ---------------- HELPERS ----------------
    def normalize_column_name(name):
        return "".join(ch for ch in str(name).lower().strip() if ch.isalnum())

    def find_column_mapping(uploaded_columns):
        norm_to_col = {normalize_column_name(c): c for c in uploaded_columns}
        mapping, used = {}, set()
        for req in REQUIRED_RAW_COLUMNS:
            for name in [req] + COLUMN_ALIASES.get(req, []):
                cand = norm_to_col.get(normalize_column_name(name))
                if cand is not None and cand not in used:
                    mapping[req] = cand
                    used.add(cand)
                    break
        for req in REQUIRED_RAW_COLUMNS:
            if req in mapping:
                continue
            free = {n: c for n, c in norm_to_col.items() if c not in used}
            hit = difflib.get_close_matches(normalize_column_name(req), list(free), n=1, cutoff=0.85)
            if hit:
                mapping[req] = free[hit[0]]
                used.add(free[hit[0]])
        return mapping

    def parse_dependents(series):
        cleaned = series.astype(str).str.replace("+", "", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")

    @st.cache_data
    def get_training_defaults():
        defaults = dict(FALLBACK_DEFAULTS)
        try:
            ref = load_raw_data()
            for c in NONCRITICAL_COLUMNS:
                if c not in ref.columns:
                    continue
                s = ref[c].dropna()
                if s.empty:
                    continue
                if c == "Dependents":
                    d = parse_dependents(s).median()
                    defaults[c] = int(d) if pd.notna(d) else 0
                elif c in NUMERIC_COLUMNS:
                    defaults[c] = float(pd.to_numeric(s, errors="coerce").median())
                else:
                    mode_val = s.mode().iloc[0]
                    if mode_val in TARGET_OPTIONS.get(c, [mode_val]):
                        defaults[c] = mode_val
        except Exception:
            pass
        return defaults

    @st.cache_data
    def get_training_medians():
        medians = {}
        try:
            ref = load_raw_data()
            for c in NUMERIC_COLUMNS:
                s = pd.to_numeric(ref[c], errors="coerce")
                s = s[s > 0]
                medians[c] = float(s.median()) if len(s) else None
        except Exception:
            pass
        return medians

    def best_scale(ratio):
        best_label, best_err = "No change", abs(np.log(ratio))
        for label, factor in SCALE_OPTIONS.items():
            err = abs(np.log(ratio * factor))
            if err < best_err - 1e-9:
                best_label, best_err = label, err
        return best_label

    def blocked(message):
        st.error(message)
        st.button("Run bulk prediction", type="primary", width="stretch",
                  disabled=True, key="bulk_predict_disabled")
        st.caption("Fix the issue above to enable prediction.")
        st.stop()

    training_defaults = get_training_defaults()

    # ---------------- UPLOAD + TEMPLATE ----------------
    section_header("Upload applicant data")
    up_col, tpl_col = st.columns([3, 1])
    with up_col:
        uploaded_file = st.file_uploader("Upload CSV, Excel or JSON file", type=["csv", "xlsx", "json"],
                                         help="CSV, Excel (.xlsx) or JSON. JSON can be a list of records, "
                                              "e.g. [{\"Gender\": \"Male\", ...}, ...].")
    with tpl_col:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        sample_data = pd.DataFrame({
            "Gender": ["Male", "Female"], "Dependents": ["0", "1"],
            "Education": ["Graduate", "Graduate"], "Self_Employed": ["No", "Yes"],
            "ApplicantIncome": [5000, 3500], "CoapplicantIncome": [1500, 1000],
            "LoanAmount": [150, 120], "Loan_Amount_Term": [360, 360],
            "Credit_History": [1, 1], "Property_Area": ["Urban", "Semiurban"],
        })
        st.download_button("Download sample CSV", data=sample_data.to_csv(index=False),
                           file_name="loan_prediction_template.csv", mime="text/csv", width="stretch")
        st.download_button("Download sample JSON", data=sample_data.to_json(orient="records", indent=2),
                           file_name="loan_prediction_template.json", mime="application/json", width="stretch")

    if uploaded_file is None:
        st.info("Upload a file to get started. Column names don't need to match exactly.")
        st.stop()

    # ---------------- READ FILE ----------------
    try:
        with st.spinner("Reading file..."):
            fname = uploaded_file.name.lower()
            if fname.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            elif fname.endswith(".json"):
                try:
                    raw_df = pd.read_json(uploaded_file, convert_dates=False)
                except ValueError:                      # e.g. {"data": [...]} or a single record
                    uploaded_file.seek(0)
                    data = json.load(uploaded_file)
                    if isinstance(data, dict):
                        lists = [v for v in data.values() if isinstance(v, list)]
                        data = lists[0] if lists else [data]
                    raw_df = pd.json_normalize(data)
            else:
                raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"We couldn't read that file. Please check it is a valid CSV, Excel or JSON file. ({e})")
        st.stop()

    if raw_df.empty:
        st.error("The uploaded file is empty.")
        st.stop()

    raw_df = raw_df.reset_index(drop=True)
    uploaded_columns = raw_df.columns.tolist()
    fid = f"{uploaded_file.name}_{uploaded_file.size}"

    st.success(f"File uploaded: {len(raw_df)} applicants found.")
    st.caption("Preview of the first 5 rows")
    st.dataframe(raw_df.head(5), width="stretch", hide_index=True)
    with st.expander("View full uploaded data", expanded=False):
        st.dataframe(raw_df, width="stretch", height=300)

    # ---------------- COLUMN MAPPING ----------------
    section_header("Validation report", "We match your columns to the 10 inputs the model needs.")
    summary_box = st.container()

    auto_mapping = find_column_mapping(uploaded_columns)
    mapping = dict(auto_mapping)
    unmatched = [c for c in REQUIRED_RAW_COLUMNS if c not in auto_mapping]

    with st.expander("Review / fix column mapping", expanded=bool(unmatched)):
        if unmatched:
            st.caption("We couldn't match these fields automatically. If your file has the data "
                       "under another name, pick that column.")
            grid = st.columns(2)
            for i, col in enumerate(unmatched):
                with grid[i % 2]:
                    choice = st.selectbox(col, [NOT_IN_FILE] + uploaded_columns, key=f"bulk_map_{fid}_{col}")
                    if choice != NOT_IN_FILE:
                        mapping[col] = choice

        table_rows = []
        for col in REQUIRED_RAW_COLUMNS:
            status = "Matched (auto)" if col in auto_mapping else ("Matched (manual)" if col in mapping else "Missing")
            table_rows.append({"Model field": col, "Mapped column": mapping.get(col, "-"), "Status": status})
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    used_cols = list(mapping.values())
    dupes = sorted({c for c in used_cols if used_cols.count(c) > 1})
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in mapping]

    with summary_box:
        matched = len(REQUIRED_RAW_COLUMNS) - len(missing)
        text = f"**{matched} of {len(REQUIRED_RAW_COLUMNS)}** required fields matched."
        if missing:
            st.error(text + f" **{len(missing)}** missing.")
        else:
            st.success(text)

    if dupes:
        blocked("The same uploaded column is mapped to more than one field: " + ", ".join(dupes))

    if missing:
        blocked(
            f"Prediction blocked: {len(missing)} required column(s) not found: {', '.join(missing)}. "
            f"The model needs all {len(REQUIRED_RAW_COLUMNS)} inputs, and filling a whole column with "
            "guessed values gives wrong predictions. If your file has the data under another name, pick it "
            "in the mapping above. Otherwise add the column and upload again (see the sample template)."
        )

    # ---------------- CREDIT SCORE -> CREDIT_HISTORY ----------------
    credit_threshold = None
    ch_num = pd.to_numeric(raw_df[mapping["Credit_History"]], errors="coerce")
    if ch_num.notna().mean() > 0.5 and ch_num.max() > 1:
        lo, hi = int(ch_num.min()), int(ch_num.max())
        with st.expander("Credit score conversion", expanded=True):
            st.info(f"`{mapping['Credit_History']}` looks like a credit score ({lo} to {hi}), but the model "
                    "needs Credit_History as 1 (good) or 0 (bad). Choose the cut-off:")
            if lo < hi:
                credit_threshold = st.slider("Scores at or above this value count as good credit (1)",
                                             lo, hi, int(min(max(650, lo), hi)), key=f"bulk_credit_{fid}")
            else:
                credit_threshold = lo

    # ---------------- BUILD STANDARDISED DATAFRAME ----------------
    df = pd.DataFrame(index=raw_df.index)
    for col in REQUIRED_RAW_COLUMNS:
        df[col] = raw_df[mapping[col]].copy()

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col].astype(str).str.strip()
                       .replace({"nan": np.nan, "None": np.nan, "NaN": np.nan, "": np.nan}))

    # ---------------- CATEGORICAL VALUES ----------------
    unresolved, value_maps_used = [], {}
    for col in CATEGORICAL_COLUMNS:
        cleaned = df[col].astype(str).str.strip().str.lower()
        was_na = df[col].isna()
        mapped = cleaned.map(BASE_MAPS[col])
        unknown_vals = sorted(cleaned[mapped.isna() & ~was_na].unique())

        if unknown_vals:
            user_map = {}
            with st.expander(f"Fix values in {col}", expanded=True):
                st.caption(f"These values in `{mapping[col]}` weren't recognised. Tell us what each one means:")
                grid = st.columns(2)
                for i, val in enumerate(unknown_vals):
                    with grid[i % 2]:
                        pick = st.selectbox(f"`{val}` means", TARGET_OPTIONS[col], index=None,
                                            placeholder="Select...", key=f"bulk_val_{fid}_{col}_{val}")
                        if pick is not None:
                            user_map[val] = pick
            value_maps_used[col] = user_map
            mapped = cleaned.map({**BASE_MAPS[col], **user_map})
            if any(v not in user_map for v in unknown_vals):
                unresolved.append(col)

        mapped[was_na] = np.nan
        df[col] = mapped

    if unresolved:
        blocked("Please tell us what the unrecognised values mean in: " + ", ".join(unresolved))

    # ---------------- NUMERIC VALUES ----------------
    df["Dependents"] = parse_dependents(df["Dependents"])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    all_empty = [c for c in df.columns if df[c].isna().all()]
    if all_empty:
        blocked("These columns have no usable values (empty or not in the expected format): "
                f"{', '.join(all_empty)}.")

    num = pd.to_numeric(df["Credit_History"], errors="coerce")
    if credit_threshold is not None:
        df["Credit_History"] = (num >= credit_threshold).astype(float).where(num.notna())
    else:
        text_ch = df["Credit_History"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0})
        df["Credit_History"] = num.where(num.isin([0, 1])).fillna(text_ch)

    # ---------------- UNIT CHECK ----------------
    train_med = get_training_medians()
    scale_choice, flagged = {}, {}
    for col in NUMERIC_COLUMNS:
        pos = df[col][df[col] > 0]
        t = train_med.get(col)
        if len(pos) and t:
            ratio = float(pos.median()) / t
            if ratio > 5 or ratio < 0.2:
                flagged[col] = ratio

    if flagged:
        with st.expander("Unit check", expanded=True):
            st.warning("Some columns look like they use different units than the data the model was "
                       "trained on. A suggested fix is pre-selected. Change it if it is wrong.")
            labels = list(SCALE_OPTIONS)
            for col, ratio in flagged.items():
                choice = st.selectbox(
                    f"{col}  (typical value in your file is {ratio:.1f}× the training data)",
                    labels, index=labels.index(best_scale(ratio)), key=f"bulk_scale_{fid}_{col}")
                scale_choice[col] = choice
                df[col] = df[col] * SCALE_OPTIONS[choice]

    # ---------------- ROW VALIDITY + FILLING ----------------
    valid_mask = ~df[CRITICAL_COLUMNS].isna().any(axis=1)
    imputed_per_row = df[NONCRITICAL_COLUMNS].isna().sum(axis=1)

    for col in NONCRITICAL_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(training_defaults[col])

    n_total = len(df)
    n_valid = int(valid_mask.sum())
    n_skipped = n_total - n_valid

    if n_valid == 0:
        blocked("No applicant has all of Credit_History, ApplicantIncome and LoanAmount filled in "
                "with valid values, so nothing can be predicted.")

    # ---------------- FEATURE ENGINEERING ----------------
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
    df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
    df["Property_Area_Semiurban"] = df["Property_Area"].eq("Semiurban").astype(int)
    df["Property_Area_Urban"] = df["Property_Area"].eq("Urban").astype(int)

    X_valid = df.loc[valid_mask, FEATURE_NAMES].copy()

    if X_valid.isna().sum().sum() > 0:
        st.dataframe(X_valid[X_valid.isna().any(axis=1)], width="stretch")
        blocked("Some processed values are still missing, so prediction has been stopped.")

    rel_valid = np.where(imputed_per_row[valid_mask] == 0, "High",
                         np.where(imputed_per_row[valid_mask] <= 2, "Medium", "Low"))

    st.success(f"Data preprocessing completed. {n_valid} of {n_total} applicants are ready for prediction.")
    if n_skipped:
        st.warning(f"{n_skipped} applicant(s) have an empty or invalid Credit_History, ApplicantIncome or "
                   "LoanAmount. They will be skipped (shown as 'Not predicted') instead of being guessed.")
    n_filled = int((imputed_per_row[valid_mask] > 0).sum())
    if n_filled:
        st.info(f"{n_filled} applicant(s) had some empty optional cells. Those were filled with typical values "
                "from the training data and marked Medium/Low in the 'Reliability' column.")

    with st.expander("View processed model features", expanded=False):
        st.dataframe(X_valid, width="stretch", height=300)

    # ---------------- RUN PREDICTION ----------------
    settings_sig = json.dumps(
        [fid, sorted(mapping.items()), credit_threshold, sorted(scale_choice.items()), value_maps_used],
        sort_keys=True, default=str)

    st.markdown("---")
    predict_bulk = st.button("Run bulk prediction", type="primary", width="stretch", key="bulk_predict_btn")

    if predict_bulk:
        n, chunk = len(X_valid), 5000
        preds, probs = [], []
        bar = st.progress(0.0, text="Predicting...")
        with st.spinner("Running the model..."):
            for start in range(0, n, chunk):
                part = X_valid.iloc[start:start + chunk]
                preds.append(rf_model.predict(part))
                probs.append(rf_model.predict_proba(part))
                bar.progress(min((start + chunk) / n, 1.0), text="Predicting...")
        bar.empty()

        predictions, probabilities = np.concatenate(preds), np.vstack(probs)
        results = raw_df.copy()
        results["Prediction"] = "Not predicted"
        results["Confidence (%)"] = np.nan
        results["Reliability"] = "N/A"
        results.loc[X_valid.index, "Prediction"] = np.where(predictions == 1, "Approved", "Not Approved")
        results.loc[X_valid.index, "Confidence (%)"] = (probabilities.max(axis=1) * 100).round(1)
        results.loc[X_valid.index, "Reliability"] = rel_valid
        st.session_state["bulk_results"] = {"sig": settings_sig, "df": results}

    # ---------------- RESULTS ----------------
    saved = st.session_state.get("bulk_results")
    if saved and saved["sig"] != settings_sig:
        st.info("The file or settings changed since the last run. Click **Run bulk prediction** again to refresh.")

    if saved and saved["sig"] == settings_sig:
        results_df = saved["df"]
        done = results_df[results_df["Prediction"] != "Not predicted"]
        total_n, predicted_n = len(results_df), len(done)
        approved_n = int((done["Prediction"] == "Approved").sum())
        rejected_n = predicted_n - approved_n
        approval_rate_b = approved_n / predicted_n * 100 if predicted_n else 0
        avg_conf = done["Confidence (%)"].mean() if predicted_n else 0

        section_header("Prediction summary")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total applicants", total_n)
        m2.metric("Predicted", predicted_n)
        m3.metric("Approved", approved_n)
        m4.metric("Not approved", rejected_n)
        m5.metric("Approval rate", f"{approval_rate_b:.1f}%")
        m6.metric("Avg confidence", f"{avg_conf:.1f}%")

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = go.Figure(go.Bar(x=["Approved", "Not approved"], y=[approved_n, rejected_n],
                                   marker=dict(color=[APPROVED, REJECTED]), text=[approved_n, rejected_n],
                                   textposition="outside", width=0.45, cliponaxis=False))
            fig.update_layout(showlegend=False,
                              yaxis=dict(title="Applicants", range=[0, max(approved_n, rejected_n, 1) * 1.2]),
                              margin=dict(t=10, b=40, l=50, r=20))
            insight_chart(fig, f"{approved_n} of {predicted_n} applicants are likely approved",
                          "Predicted outcomes for this file.", height=280)
        with ch2:
            fig = go.Figure(go.Histogram(x=done["Confidence (%)"], nbinsx=20, marker=dict(color=ACCENT)))
            fig.update_layout(showlegend=False, xaxis=dict(title="Confidence (%)"),
                              yaxis=dict(title="Applicants"), margin=dict(t=10, b=40, l=50, r=20))
            insight_chart(fig, f"Average model confidence is {avg_conf:.0f}%",
                          "How sure the model is across all predicted applicants.", height=280)

        section_header("Prediction results")
        f1_, f2_, f3_ = st.columns(3)
        with f1_:
            pred_filter = st.multiselect("Prediction", ["Approved", "Not Approved", "Not predicted"],
                                         default=["Approved", "Not Approved", "Not predicted"])
        with f2_:
            rel_filter = st.multiselect("Reliability", ["High", "Medium", "Low", "N/A"],
                                        default=["High", "Medium", "Low", "N/A"])
        with f3_:
            min_conf = st.slider("Minimum confidence (%)", 0, 100, 0)

        view = results_df[
            results_df["Prediction"].isin(pred_filter)
            & results_df["Reliability"].isin(rel_filter)
            & ((results_df["Confidence (%)"] >= min_conf) | results_df["Confidence (%)"].isna())
        ].sort_values("Confidence (%)", ascending=False, na_position="last")

        st.caption(f"Showing {len(view)} of {total_n} applicants.")
        st.dataframe(view, width="stretch", height=450)
        dl1, dl2 = st.columns(2)
        dl1.download_button("Download results as CSV", data=view.to_csv(index=False),
                            file_name="loan_bulk_prediction_results.csv", mime="text/csv", width="stretch")
        dl2.download_button("Download results as JSON", data=view.to_json(orient="records", indent=2),
                            file_name="loan_bulk_prediction_results.json", mime="application/json", width="stretch")

        with st.expander("Check accuracy against the actual outcome (optional)", expanded=False):
            st.caption("If your file already has the real result (approved / not approved), pick that "
                       "column to see how well the model did.")
            original_cols = list(raw_df.columns)
            guess = next((c for c in original_cols if normalize_column_name(c) in LABEL_HINTS), None)
            options = [NO_LABEL] + original_cols
            label_col = st.selectbox("Actual outcome column", options,
                                     index=options.index(guess) if guess else 0, key=f"bulk_label_{fid}")

            if label_col != NO_LABEL:
                actual_text = results_df[label_col].astype(str).str.strip().str.lower()
                actual = actual_text.map(
                    lambda v: 1 if v in POSITIVE_WORDS else (0 if v in NEGATIVE_WORDS else np.nan))
                predicted = results_df["Prediction"].map({"Approved": 1, "Not Approved": 0})
                ok = actual.notna() & predicted.notna()

                if ok.sum() == 0:
                    st.warning(f"Could not read `{label_col}` as approved / not approved values, "
                               "so accuracy can't be calculated.")
                else:
                    y_true, y_pred = actual[ok].astype(int), predicted[ok].astype(int)
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Accuracy", f"{accuracy_score(y_true, y_pred) * 100:.1f}%")
                    a2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0) * 100:.1f}%")
                    a3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0) * 100:.1f}%")
                    a4.metric("F1 score", f"{f1_score(y_true, y_pred, zero_division=0) * 100:.1f}%")
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    st.dataframe(pd.DataFrame(
                        cm, index=["Actual: Not approved", "Actual: Approved"],
                        columns=["Predicted: Not approved", "Predicted: Approved"]), width="stretch")
                    st.caption(f"Based on {int(ok.sum())} applicants that have both a prediction "
                               "and a readable actual outcome.")


# # ==================================================================
# # PAGE: MODEL PERFORMANCE
# # ==================================================================
# elif page == "Model Performance":
#     page_header("Model Performance", "How the three candidate models compare on unseen data.")
#     rows, extra = evaluate_models()
#     best = max(range(len(rows)), key=lambda i: rows[i]["Accuracy"])

#     with st.container(border=True):
#         section_header("Model comparison", "Same models and same 20% hold-out split as your notebook. Best accuracy is highlighted.")
#         trs = [[r["Model"]] + [f"{r[k]*100:.1f}%" for k in ["Accuracy", "Precision", "Recall", "F1"]]
#                + [f"{r['ROC-AUC']:.2f}"] for r in rows]
#         st.markdown(html_table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], trs,
#                                best_idx=best, num_cols=(1, 2, 3, 4, 5)), unsafe_allow_html=True)
#         st.markdown('<div class="sec-sub" style="margin-top:10px">'
#                     'The dataset is imbalanced (about 2.2 approved for every 1 rejected), so accuracy alone '
#                     'can look good even for a weak model. Check precision, recall and F1 too.</div>',
#                     unsafe_allow_html=True)

#     c1, c2 = st.columns(2)
#     with c1:
#         cm = extra["cm"]
#         labels = ["Not approved", "Approved"]
#         fig = go.Figure(go.Heatmap(
#             z=cm, x=labels, y=labels, colorscale=[[0, INPUT], [1, ACCENT]], showscale=False,
#             text=cm, texttemplate="%{text}", textfont=dict(size=18, color=TEXT),
#             hovertemplate="Actual %{y}, predicted %{x}: %{z}<extra></extra>"))
#         fig.update_layout(xaxis=dict(title="Predicted"), yaxis=dict(title="Actual", autorange="reversed"),
#                           margin=dict(t=10, b=50, l=90, r=10))
#         insight_chart(fig, f"Random Forest got {int(cm[0,0] + cm[1,1])} of {int(cm.sum())} test applicants right",
#                       "Each cell counts applicants. The diagonal is correct predictions.", height=340)
#     with c2:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=extra["fpr"], y=extra["tpr"], mode="lines", line=dict(color=ACCENT, width=3),
#                                  name=f"Random Forest (AUC = {extra['auc']:.2f})"))
#         fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"),
#                                  name="Random guessing"))
#         fig.update_layout(xaxis=dict(title="False positive rate", range=[0, 1]),
#                           yaxis=dict(title="True positive rate", range=[0, 1.02]),
#                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
#                           margin=dict(t=30, b=50, l=60, r=10))
#         insight_chart(fig, f"The model separates approved from rejected well (AUC {extra['auc']:.2f})",
#                       "The further the curve is above the dashed line, the better.", height=340)

#     imp = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False).head(10)
#     imp = imp.sort_values(ascending=True)
#     fig = go.Figure(go.Bar(x=imp.values * 100, y=[FRIENDLY[i] for i in imp.index], orientation="h",
#                            marker=dict(color=ACCENT), text=[f"{v*100:.1f}%" for v in imp.values],
#                            textposition="outside", cliponaxis=False))
#     fig.update_layout(showlegend=False, xaxis=dict(title="Importance (%)", range=[0, imp.max() * 100 * 1.2]),
#                       yaxis=dict(title=""), margin=dict(t=10, b=40, l=10, r=40))
#     insight_chart(fig, f"{FRIENDLY[imp.index[-1]]} is the biggest driver of predictions",
#                   "Top 10 features in the deployed Random Forest model.", height=380)