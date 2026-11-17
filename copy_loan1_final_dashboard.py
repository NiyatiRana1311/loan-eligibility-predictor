import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------------
# COLOR PALETTE — White background with #9282FF accent
# ------------------------------------------------------------------
BG            = "#FFFFFF"       # Clean white background
SIDEBAR_BG    = "#F8F9FC"       # Soft light gray/lavender sidebar surface
CARD_BG       = "#FFFFFF"       # Crisp white card surface
INPUT_BG      = "#FFFFFF"       # Input surface
BORDER        = "#E2E8F0"       # Light gray border (1px)
BORDER_LIGHT  = "#CBD5E1"       # Subtle border highlight
TEXT_PRIMARY  = "#0F172A"       # Deep slate primary text
TEXT_MUTED    = "#64748B"       # Muted slate secondary text

# Primary accent & status colors
ACCENT_COLOR  = "#9282FF"       # Brand accent #9282FF
ACCENT_ORANGE = "#9282FF"       # Compatibility alias
ACCENT_HOVER  = "#7D6BF7"       # Deeper shade for hover

APPROVED      = "#9282FF"       # Approved status (#9282FF)
NOT_APPROVED  = "#64748B"       # Not approved status (Muted slate #64748B)
WARNING       = "#E0A838"       # Warning gold

# Functional compatibility aliases
GREEN         = APPROVED
RED           = NOT_APPROVED
AMBER         = WARNING
ACCENT_LIME   = ACCENT_COLOR
ACCENT_LIME_DARK = "#FFFFFF"
ACCENT_LIME_BG = "rgba(146, 130, 255, 0.12)"
GREEN_LIGHT   = "rgba(146, 130, 255, 0.12)"
GREEN_BORDER  = "#E2E8F0"
RED_LIGHT     = "rgba(100, 116, 139, 0.12)"
RED_BORDER    = "#E2E8F0"
AMBER_LIGHT   = "rgba(224, 168, 56, 0.12)"
AMBER_BORDER  = "#E2E8F0"
PURPLE_DARK   = ACCENT_COLOR
PURPLE        = ACCENT_COLOR
PURPLE_LIGHT  = "rgba(146, 130, 255, 0.12)"
GRID_COLOR    = "#F1F5F9"       # Soft light grid lines

# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
# scaler.pkl isn't needed for Random Forest predictions (RF was trained on
# unscaled features in the notebook) — kept loaded only if you reintroduce
# a scaled model later.
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURE_NAMES = [
    "Gender", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
    "Property_Area_Semiurban", "Property_Area_Urban", "TotalIncome"
]

DATA_FILE = "Copy_of_loan.xlsx"  # your raw training dataset, kept alongside this file

@st.cache_data
def load_raw_data():
    return pd.read_excel(DATA_FILE)

st.set_page_config(
    page_title="Loan Eligibility Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# LOAD EXTERNAL CSS (UTF-8, INJECTED ONCE)
# ------------------------------------------------------------------
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
    elif Path("styles.css").exists():
        css = Path("styles.css").read_text(encoding="utf-8")
    else:
        css = ""
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "page" in st.query_params:
    st.session_state.page = st.query_params["page"]
    st.query_params.clear()
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: result, confidence, income, loan_amount, time

# ------------------------------------------------------------------
# SIDEBAR (FIXED WIDTH DEFINED IN STYLES.CSS)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Loan Predictor")
    st.caption("Predict loan eligibility using machine learning.")
    st.markdown("---")

    if st.button("Overview", width="stretch", key="nav_btn_ov",
                 type="primary" if st.session_state.page == "Overview" else "secondary"):
        st.query_params.clear()
        st.session_state.page = "Overview"
        st.rerun()

    if st.button("Data Overview", width="stretch", key="nav_btn_ad",
                 type="primary" if st.session_state.page == "Data Overview" else "secondary"):
        st.query_params.clear()
        st.session_state.page = "Data Overview"
        st.rerun()

    if st.button("Data Insights", width="stretch", key="nav_btn_di",
                 type="primary" if st.session_state.page == "Data_Insights" else "secondary"):
        st.query_params.clear()
        st.session_state.page = "Data_Insights"
        st.rerun()

    if st.button("Loan Form", width="stretch", key="nav_btn_lf",
                 type="primary" if st.session_state.page == "Loan_Form" else "secondary"):
        st.query_params.clear()
        st.session_state.page = "Loan_Form"
        st.rerun()

    if st.button("Bulk Prediction", width="stretch", key="nav_btn_bp",
                 type="primary" if st.session_state.page == "Bulk_Prediction" else "secondary"):
        st.query_params.clear()
        st.session_state.page = "Bulk_Prediction"
        st.rerun()

    st.markdown("---")
    st.markdown("**Model Used**")
    st.markdown("- Random Forest (Best Accuracy)")

    st.markdown("---")
    st.caption("Machine Learning Project")

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def render_page_header(title, subtitle, pills=None):
    pills_html = ""
    if pills:
        for item in pills:
            text = item[0]
            bg = item[1] if len(item) > 1 else "rgba(146, 130, 255, 0.12)"
            color = item[2] if len(item) > 2 else ACCENT_COLOR
            border = item[3] if len(item) > 3 else BORDER
            pills_html += (
                f'<span class="header-pill" style="background:{bg}; color:{color}; border:1px solid {border};">'
                f'{text}</span>'
            )
    header_html = (
        f'<div style="display:flex; align-items:center; justify-content:space-between; '
        f'margin-bottom:20px; padding-bottom:14px; border-bottom:1px solid {BORDER}; flex-wrap:wrap; gap:12px;">'
        f'<div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="section-subtitle" style="margin-top:4px; margin-bottom:0;">{subtitle}</div>'
        f'</div>'
        f'<div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">{pills_html}</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def metric_card(icon, icon_bg, label, value, sub=None, positive=True):
    sub_html = ""
    if sub:
        cls = "metric-sub-positive" if positive else "metric-sub-negative"
        arrow = "▲" if positive else "▼"
        sub_html = f'<div class="{cls}">{arrow} {sub}</div>'
    card_html = (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def progress_row(label, value_text, pct, color):
    st.markdown(f"""
        <div class="progress-row">
            <div class="progress-label"><span>{label}</span><span>{value_text}</span></div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=13),
        margin=dict(t=40, b=30, l=10, r=10),
        height=height,
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    )
    return fig


def bullet_bar(value, label, color, baseline=None, height=180):
    """A slim horizontal bullet chart with an optional baseline reference tick."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[value], y=[label], orientation="h",
        marker=dict(color=color, cornerradius=6),
        text=[f"{value:.1f}%"], textposition="outside", cliponaxis=False,
        width=0.5,
    ))
    if baseline is not None:
        fig.add_shape(
            type="line", x0=baseline, x1=baseline, y0=-0.4, y1=0.4,
            line=dict(color=TEXT_MUTED, width=2, dash="dash"),
        )
        fig.add_annotation(
            x=baseline, y=0.55, text=f"Baseline {baseline:.1f}%",
            showarrow=False, font=dict(size=11, color=TEXT_MUTED),
        )
    fig = style_fig(fig, height=height)
    fig.update_layout(
        xaxis=dict(range=[0, 115], title="Accuracy (%)", gridcolor=GRID_COLOR),
        yaxis=dict(title=""),
        showlegend=False,
        margin=dict(t=40, b=30, l=10, r=40),
    )
    return fig


def donut_with_center(labels, values, colors, center_text, center_sub, height=300):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.72,
        marker=dict(colors=colors, line=dict(color=CARD_BG, width=2)),
        textinfo="none",
    ))
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        showlegend=False,
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[
            dict(text=f"<b>{center_text}</b>", x=0.5, y=0.55, font=dict(family="Inter, sans-serif", size=28, color=ACCENT_COLOR), showarrow=False),
            dict(text=center_sub, x=0.5, y=0.42, font=dict(family="Inter, sans-serif", size=12, color=TEXT_MUTED), showarrow=False),
        ]
    )
    return fig


# ==================================================================
# PAGE 1: OVERVIEW
# ==================================================================
if st.session_state.page == "Overview":

    today = datetime.now().strftime("%A, %d %b %Y")

    # ---- Page header row ----
    render_page_header(
        "Overview",
        f"Loan Eligibility Predictor &nbsp;&bull;&nbsp; {today}",
        [
            ("● Live", "rgba(146, 130, 255, 0.12)", ACCENT_COLOR, BORDER),
            ("Random Forest", "rgba(146, 130, 255, 0.12)", ACCENT_COLOR, BORDER),
            ("Accuracy: 84.6%", "rgba(224, 168, 56, 0.15)", WARNING, BORDER),
        ]
    )

    # ---- Stat cards row ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Model Accuracy</div>'
            f'<div class="metric-value">84.6%</div>'
            f'<div class="metric-sub-positive">▲ vs baseline</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Dataset Size</div>'
            f'<div class="metric-value">614</div>'
            f'<div class="metric-sub-positive">▲ training records</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Features Used</div>'
            f'<div class="metric-value">12</div>'
            f'<div class="metric-sub-positive">▲ engineered inputs</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Best Model</div>'
            f'<div class="metric-value" style="font-size:20px; line-height:1.2;">Random Forest</div>'
            f'<div class="metric-sub-positive">▲ of 3 models</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # Compute actual approval rate from dataset
    try:
        raw_ov = load_raw_data()
        total_rec = len(raw_ov)
        appr_rec = int((raw_ov["Loan_Status"] == "Y").sum())
        appr_pct = (appr_rec / total_rec) * 100 if total_rec > 0 else 68.7
    except Exception:
        appr_pct = 68.7

    c1, c2, c3 = st.columns(3, gap="medium")

    # ---- 1. APPROVAL SPLIT CARD (Flat Card) ----
    with c1:
        dash_array = 226.2
        dash_offset = dash_array * (1 - (appr_pct / 100.0))
        donut_html = (
            f'<div class="flat-card" style="min-height:226px; display:flex; flex-direction:column; justify-content:space-between;">'
            f'<div class="section-title" style="margin:0;">Approval split</div>'
            f'<div style="display:flex; justify-content:center; align-items:center; margin:10px 0;">'
            f'<svg width="124" height="124" viewBox="0 0 100 100">'
            f'<circle cx="50" cy="50" r="36" fill="none" stroke="{GRID_COLOR}" stroke-width="12" />'
            f'<circle cx="50" cy="50" r="36" fill="none" stroke="{ACCENT_COLOR}" stroke-width="12" '
            f'stroke-dasharray="{dash_array}" stroke-dashoffset="{dash_offset:.1f}" '
            f'stroke-linecap="round" transform="rotate(-90 50 50)" />'
            f'<text x="50" y="55" text-anchor="middle" font-family="Inter, sans-serif" font-size="16.5" font-weight="700" fill="{TEXT_PRIMARY}">'
            f'{appr_pct:.1f}%</text>'
            f'</svg>'
            f'</div>'
            f'<div style="text-align:center; font-size:13px; font-weight:500; color:{TEXT_MUTED};">'
            f'approved in dataset'
            f'</div>'
            f'</div>'
        )
        st.markdown(donut_html, unsafe_allow_html=True)

    # ---- 2. CHECK YOUR ELIGIBILITY CTA CARD (Flat Card) ----
    with c2:
        st.markdown(
            f'<div class="flat-card" style="min-height:226px; display:flex; flex-direction:column; justify-content:space-between;">'
            f'<div>'
            f'<div class="section-title">Check your eligibility</div>'
            f'<div class="section-subtitle">Takes under a minute. Instant credit risk evaluation based on the Random Forest model.</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("Open loan form", key="cta_open_loan_form", width="stretch", type="primary"):
            st.query_params.clear()
            st.session_state.page = "Loan_Form"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 3. RECENT PREDICTIONS CARD (Flat Card) ----
    with c3:
        if "history" not in st.session_state or not st.session_state.history:
            st.session_state.history = [
                {"time": "10:42", "result": "Approved", "confidence": 91},
                {"time": "10:45", "result": "Not approved", "confidence": 78},
                {"time": "10:51", "result": "Approved", "confidence": 84},
            ]

        # Always take the 3 latest predictions, newest on top
        recent_list = st.session_state.history[-3:][::-1]

        rows_html = ""
        for item in recent_list:
            t = str(item.get("time", "12:00"))[:5]
            res = str(item.get("result", "Approved"))
            conf = float(item.get("confidence", 85))
            is_app = ("approved" in res.lower() and "not" not in res.lower())
            badge_bg = "rgba(146, 130, 255, 0.12)" if is_app else "rgba(100, 116, 139, 0.1)"
            badge_col = APPROVED if is_app else NOT_APPROVED
            status_text = "Approved" if is_app else "Not approved"
            rows_html += (
                f'<div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid {BORDER};">'
                f'<span style="font-size:13px; font-weight:500; color:{TEXT_PRIMARY};">{t}</span>'
                f'<span style="background:{badge_bg}; color:{badge_col}; font-size:12px; font-weight:600; padding:3px 10px; border-radius:6px;">{status_text}</span>'
                f'<span style="font-size:13px; font-weight:600; color:{TEXT_PRIMARY}; min-width:32px; text-align:right;">{conf:.0f}%</span>'
                f'</div>'
            )

        recent_html = (
            f'<div class="flat-card" style="min-height:226px; display:flex; flex-direction:column; justify-content:space-between;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
            f'<span class="section-title" style="margin:0;">Recent predictions</span>'
            f'<span style="font-size:11px; font-weight:600; color:{ACCENT_COLOR}; background:rgba(146, 130, 255, 0.12); border:1px solid {BORDER}; padding:2px 8px; border-radius:6px;">LIVE</span>'
            f'</div>'
            f'<div>{rows_html}</div>'
            f'</div>'
        )
        st.markdown(recent_html, unsafe_allow_html=True)

    # ---- FULL-WIDTH "HOW IT WORKS" FLAT CARD ----
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

    steps_data = [
        ("01", "Enter Your Details", "Fill in income, loan term & personal info in the form.", "→"),
        ("02", "We Process Data", "Inputs are normalized, feature-engineered & prepared.", "→"),
        ("03", "AI Model Predicts", "Trained Random Forest classifier evaluates credit risk.", "→"),
        ("04", "Instant Decision", "View approval status with calibrated confidence score.", "✓"),
    ]

    steps_cols_html = "".join([
        f'<div style="background:{SIDEBAR_BG}; border:1px solid {BORDER}; border-radius:8px; padding:18px 20px; '
        f'display:flex; flex-direction:column; justify-content:space-between;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">'
        f'<span style="width:28px; height:28px; border-radius:6px; background:#FFFFFF; color:{ACCENT_COLOR}; '
        f'display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; '
        f'border:1px solid {BORDER};">{num}</span>'
        f'<span style="color:{ACCENT_COLOR}; font-size:15px; font-weight:600;">{arrow}</span>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:14px; font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:4px;">{title}</div>'
        f'<div style="font-size:13px; color:{TEXT_MUTED}; line-height:1.5;">{desc}</div>'
        f'</div>'
        f'</div>'
        for num, title, desc, arrow in steps_data
    ])

    how_full_html = (
        f'<div class="flat-card">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">'
        f'<div>'
        f'<div class="section-title" style="margin:0;">How It Works</div>'
        f'<div class="section-subtitle" style="margin:2px 0 0 0;">Check your loan eligibility in four simple automated steps.</div>'
        f'</div>'
        f'<span style="background:rgba(146, 130, 255, 0.12); color:{ACCENT_COLOR}; font-size:12px; font-weight:600; padding:4px 12px; border-radius:6px; border:1px solid rgba(146, 130, 255, 0.25);">4-Step Pipeline</span>'
        f'</div>'
        f'<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:14px;">'
        f'{steps_cols_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(how_full_html, unsafe_allow_html=True)


# ==================================================================
# PAGE 2: DATA OVERVIEW
# ==================================================================
elif st.session_state.page == "Data Overview":

    try:
        raw = load_raw_data()
    except FileNotFoundError:
        st.error(f"Couldn't find `{DATA_FILE}`. Place it in the same folder as this file.")
        st.stop()

    total_rows = len(raw)
    total_columns = len(raw.columns)
    total_cells = total_rows * total_columns

    missing_by_col = raw.isnull().sum()
    total_missing = int(missing_by_col.sum())
    completeness = ((total_cells - total_missing) / total_cells) * 100
    duplicate_count = int(raw.duplicated().sum())
    approved_count = int((raw["Loan_Status"] == "Y").sum())
    rejected_count = int((raw["Loan_Status"] == "N").sum())

    if rejected_count > 0:
        class_ratio = approved_count / rejected_count
    else:
        class_ratio = 0

    render_page_header(
        "Data Overview",
        "Loan prediction dataset &nbsp;&bull;&nbsp; Columns, types and data health",
        [
            (f"{total_rows} Records", "rgba(146, 130, 255, 0.12)", ACCENT_COLOR, BORDER),
            (f"{total_columns} Columns", "rgba(146, 130, 255, 0.12)", ACCENT_COLOR, BORDER),
            (f"{completeness:.1f}% Complete", "rgba(224, 168, 56, 0.15)", WARNING, BORDER),
        ]
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("N", "rgba(146, 130, 255, 0.12)", "Rows", f"{total_rows}", "applications", positive=True)
    with m2:
        metric_card("C", "rgba(146, 130, 255, 0.12)", "Columns", f"{total_columns}", "incl. ID and target", positive=True)
    with m3:
        metric_card("%", "rgba(146, 130, 255, 0.12)", "Completeness", f"{completeness:.1f}%",
                    f"{total_missing:,} empty cells", positive=True)
    with m4:
        metric_card("D", "rgba(146, 130, 255, 0.12)", "Duplicates", f"{duplicate_count}",
                    "none found" if duplicate_count == 0 else "found",
                    positive=duplicate_count == 0)
    with m5:
        metric_card("R", "rgba(224, 168, 56, 0.15)", "Class Balance", f"{class_ratio:.1f} : 1",
                    "approved to rejected", positive=True)

    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.55, 1])

    # ---------------- DATA DICTIONARY ----------------
    with left_col:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Data dictionary</div>'
                '<div class="section-subtitle">'
                'Every column, its type and how many values are missing'
                '</div>',
                unsafe_allow_html=True
            )

            dictionary_rows = []

            for col in raw.columns:
                if col == "Loan_ID":
                    data_type = "ID"
                    type_class = "type-id"
                elif col == "Loan_Status":
                    data_type = "Target"
                    type_class = "type-target"
                elif col == "Credit_History":
                    # Credit_History is categorical (0/1)
                    data_type = "Category"
                    type_class = "type-category"
                elif pd.api.types.is_numeric_dtype(raw[col]):
                    data_type = "Number"
                    type_class = "type-number"
                else:
                    data_type = "Category"
                    type_class = "type-category"

                meanings = {
                    "Loan_ID": "Unique application number",
                    "Gender": "Male or Female",
                    "Married": "Marital status",
                    "Dependents": "0, 1, 2 or 3+",
                    "Education": "Graduate or Not Graduate",
                    "Self_Employed": "Yes or No",
                    "ApplicantIncome": "Applicant monthly income",
                    "CoapplicantIncome": "Coapplicant monthly income",
                    "LoanAmount": "Requested amount (thousands)",
                    "Loan_Amount_Term": "Repayment term in months",
                    "Credit_History": "1 = bills paid on time, 0 = otherwise",
                    "Property_Area": "Urban, Semiurban or Rural",
                    "Loan_Status": "Approved (Y) or not (N)"
                }

                meaning = meanings.get(col, "Dataset variable")
                missing = int(raw[col].isnull().sum())

                dictionary_rows.append({
                    "Column": col,
                    "Type": data_type,
                    "TypeClass": type_class,
                    "Meaning": meaning,
                    "Missing": missing
                })

            h1, h2, h3, h4 = st.columns([1.05, 0.8, 1.65, 0.45])
            with h1:
                st.markdown("**Column**")
            with h2:
                st.markdown("**Type**")
            with h3:
                st.markdown("**Meaning**")
            with h4:
                st.markdown("**Missing**")

            st.divider()

            for item in dictionary_rows:
                c1, c2, c3, c4 = st.columns([1.05, 0.8, 1.65, 0.45])
                with c1:
                    st.markdown(f'<div class="small-table">{item["Column"]}</div>',
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<span class="type-pill {item["TypeClass"]}">{item["Type"]}</span>',
                                unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="small-table">{item["Meaning"]}</div>',
                                unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="small-table">{item["Missing"]}</div>',
                                unsafe_allow_html=True)

    # ---------------- RIGHT COLUMN ----------------
    with right_col:

        # MISSING VALUES (cliponaxis=False, textposition="outside", expanded axis range)
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Missing values</div>'
                '<div class="section-subtitle">Filled with median or most common value</div>',
                unsafe_allow_html=True
            )

            missing_df = missing_by_col[missing_by_col > 0].sort_values(ascending=True)

            if len(missing_df) > 0:
                missing_fig = go.Figure(
                    go.Bar(
                        x=missing_df.values,
                        y=missing_df.index,
                        orientation="h",
                        marker=dict(
                            color=[ACCENT_COLOR if i >= len(missing_df) - 2 else "rgba(146, 130, 255, 0.65)" for i in range(len(missing_df))],
                            cornerradius=6
                        ),
                        text=missing_df.values,
                        textposition="outside",
                        cliponaxis=False,
                    )
                )
                missing_fig.update_traces(cliponaxis=False)
                missing_fig = style_fig(missing_fig, height=280)
                missing_fig.update_layout(
                    xaxis=dict(
                        title="Missing values",
                        gridcolor=GRID_COLOR,
                        range=[0, max(missing_df.values) * 1.25],
                    ),
                    yaxis=dict(title="", gridcolor=GRID_COLOR),
                    showlegend=False,
                    margin=dict(t=10, b=35, l=110, r=45)
                )
                st.plotly_chart(missing_fig, width="stretch", config={"displayModeBar": False})
            else:
                st.success("No missing values found.")

        # COLUMN TYPES (Credit_History counted as Category)
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Column types</div>'
                '<div class="section-subtitle">'
                f'How the {total_columns} columns split'
                '</div>',
                unsafe_allow_html=True
            )

            category_count = 0
            number_count = 0
            id_count = 0
            target_count = 0

            for col in raw.columns:
                if col == "Loan_ID":
                    id_count += 1
                elif col == "Loan_Status":
                    target_count += 1
                elif col == "Credit_History":
                    # Categorical (0/1)
                    category_count += 1
                elif pd.api.types.is_numeric_dtype(raw[col]):
                    number_count += 1
                else:
                    category_count += 1

            type_labels = ["Category", "Number", "ID", "Target"]
            type_values = [category_count, number_count, id_count, target_count]
            type_colors = [ACCENT_COLOR, "#94A3B8", "#64748B", WARNING]

            type_fig = go.Figure(
                go.Pie(
                    labels=type_labels,
                    values=type_values,
                    hole=0.68,
                    marker=dict(colors=type_colors, line=dict(color=CARD_BG, width=2)),
                    textinfo="none"
                )
            )
            type_fig.update_layout(
                paper_bgcolor=CARD_BG,
                plot_bgcolor=CARD_BG,
                font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=13),
                height=250,
                margin=dict(t=5, b=5, l=5, r=5),
                legend=dict(orientation="v", x=0.88, y=0.5),
                annotations=[
                    dict(text=f"<b>{total_columns}</b>", x=0.43, y=0.55,
                         font=dict(family="Inter, sans-serif", size=24, color=ACCENT_COLOR), showarrow=False),
                    dict(text="columns", x=0.43, y=0.42,
                         font=dict(family="Inter, sans-serif", size=12, color=TEXT_MUTED), showarrow=False)
                ]
            )
            st.plotly_chart(type_fig, width="stretch", config={"displayModeBar": False})

    # ---------------- NUMERIC SUMMARY + TARGET BALANCE ----------------
    bottom_left, bottom_right = st.columns([1.25, 1])

    with bottom_left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Numeric summary</div>'
                '<div class="section-subtitle">Income and loan amount statistics</div>',
                unsafe_allow_html=True
            )

            numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
            available_numeric = [c for c in numeric_cols if c in raw.columns]

            if available_numeric:
                summary = raw[available_numeric].describe().T

                summary_table = pd.DataFrame(
                    index=["Mean", "Median", "Std dev", "Min", "Max"],
                    columns=available_numeric
                )
                summary_table.loc["Mean"] = [raw[c].mean() for c in available_numeric]
                summary_table.loc["Median"] = [raw[c].median() for c in available_numeric]
                summary_table.loc["Std dev"] = [raw[c].std() for c in available_numeric]
                summary_table.loc["Min"] = [raw[c].min() for c in available_numeric]
                summary_table.loc["Max"] = [raw[c].max() for c in available_numeric]

                display_table = summary_table.copy()

                for col in display_table.columns:
                    display_table[col] = display_table[col].apply(
                        lambda x: f"{x:,.0f}" if pd.notnull(x) else "-"
                    )

                display_table.columns = [
                    "Applicant inc." if c == "ApplicantIncome"
                    else "Coapplicant inc." if c == "CoapplicantIncome"
                    else "Loan amount" if c == "LoanAmount"
                    else "Term (months)"
                    for c in display_table.columns
                ]

                st.dataframe(display_table, width="stretch", hide_index=False)

    with bottom_right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Target balance</div>'
                '<div class="section-subtitle">'
                f'Loan_Status across all {total_rows} applications'
                '</div>',
                unsafe_allow_html=True
            )

            target_fig = go.Figure()
            target_fig.add_trace(go.Bar(
                x=[approved_count], y=["Approved"], orientation="h",
                marker=dict(color=APPROVED, cornerradius=6),
                text=[approved_count], textposition="outside", cliponaxis=False
            ))
            target_fig.add_trace(go.Bar(
                x=[rejected_count], y=["Not approved"], orientation="h",
                marker=dict(color=NOT_APPROVED, cornerradius=6),
                text=[rejected_count], textposition="outside", cliponaxis=False
            ))
            target_fig.update_traces(cliponaxis=False)
            target_fig = style_fig(target_fig, height=230)
            target_fig.update_layout(
                xaxis=dict(
                    title="Applications",
                    gridcolor=GRID_COLOR,
                    range=[0, max(approved_count, rejected_count) * 1.2]
                ),
                yaxis=dict(title=""),
                showlegend=False,
                margin=dict(t=20, b=40, l=85, r=45)
            )
            st.plotly_chart(target_fig, width="stretch", config={"displayModeBar": False})

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"**<span style='color:{APPROVED}'>●</span> Approved ({approved_count})**",
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f"**<span style='color:{NOT_APPROVED}'>●</span> Not approved ({rejected_count})**",
                    unsafe_allow_html=True
                )

    # ---------------- DATA PREPARATION PIPELINE ----------------
    st.markdown("### Data preparation pipeline")

    p1, p2, p3, p4, p5 = st.columns(5)

    steps = [
        ("STEP 1", "Checked shape, types, nulls and duplicates"),
        ("STEP 2", "Filled gaps with median or mode values"),
        ("STEP 3", "Encoded text columns and one hot encoding"),
        ("STEP 4", "Added TotalIncome and prepared model features"),
        ("STEP 5", "Split train and test, scaled inputs for Random Forest")
    ]

    for col, (step, text) in zip([p1, p2, p3, p4, p5], steps):
        with col:
            with st.container(border=True):
                st.caption(step)
                st.markdown(f"**{text}**")


# ==================================================================
# PAGE 3: DATA INSIGHTS
# ==================================================================
elif st.session_state.page == "Data_Insights":

    try:
        raw = load_raw_data()
    except FileNotFoundError:
        st.error(f"Couldn't find `{DATA_FILE}`. Place it in the same folder as this file.")
        st.stop()

    total_apps = len(raw)
    approved = (raw["Loan_Status"] == "Y").sum()
    not_approved = (raw["Loan_Status"] == "N").sum()
    avg_income = raw["ApplicantIncome"].mean()
    avg_loan = raw["LoanAmount"].mean()

    render_page_header(
        "Data Insights",
        "Training data patterns &nbsp;&bull;&nbsp; Historical loan approval analysis",
        [
            (f"{total_apps} Applicants", "rgba(146, 130, 255, 0.12)", ACCENT_COLOR, BORDER),
            (f"Approval: {approved / total_apps:.1%}", "rgba(146, 130, 255, 0.12)", APPROVED, BORDER),
            ("Credit History Focus", "rgba(224, 168, 56, 0.15)", WARNING, BORDER),
        ]
    )

    kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)

    with kcol1:
        metric_card("N", "rgba(146, 130, 255, 0.12)", "Total Applications", f"{total_apps}",
                    "training sample", positive=True)
    with kcol2:
        metric_card("Y", "rgba(146, 130, 255, 0.12)", "Approved", f"{approved}",
                    f"{approved / total_apps:.1%}", positive=True)
    with kcol3:
        metric_card("X", "rgba(100, 116, 139, 0.12)", "Not Approved", f"{not_approved}",
                    f"{not_approved / total_apps:.1%}", positive=False)
    with kcol4:
        metric_card("₹", "rgba(224, 168, 56, 0.15)", "Avg Applicant Income", f"₹{avg_income:,.0f}",
                    "per applicant", positive=True)
    with kcol5:
        metric_card("₹", "rgba(146, 130, 255, 0.12)", "Avg Loan Amount", f"₹{avg_loan:,.0f}k",
                    "average requested", positive=True)

    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

    # ---------------- ROW 1 ----------------
    row1c1, row1c2 = st.columns(2)

    # Credit History vs Loan Status
    with row1c1:
        with st.container(border=True):
            st.markdown(
                '<div class="chart-title">Credit History vs Loan Status</div>'
                '<div class="chart-subtitle">Loan approval status based on credit history.</div>',
                unsafe_allow_html=True
            )

            ch = pd.crosstab(raw["Credit_History"], raw["Loan_Status"])

            if "Y" not in ch.columns:
                ch["Y"] = 0
            if "N" not in ch.columns:
                ch["N"] = 0

            ch = ch.reindex([0.0, 1.0], fill_value=0)

            ch_fig = go.Figure()
            ch_fig.add_trace(go.Bar(
                x=["No Credit History", "Has Credit History"],
                y=ch["Y"], name="Approved", marker_color=APPROVED, width=0.35,
                cliponaxis=False
            ))
            ch_fig.add_trace(go.Bar(
                x=["No Credit History", "Has Credit History"],
                y=ch["N"], name="Not Approved", marker_color=NOT_APPROVED, width=0.35,
                cliponaxis=False
            ))
            ch_fig.update_traces(marker_cornerradius=6, cliponaxis=False)

            ch_fig = style_fig(ch_fig, height=245)
            ch_fig.update_layout(
                barmode="group",
                yaxis=dict(title="Count", gridcolor=GRID_COLOR),
                xaxis=dict(gridcolor=GRID_COLOR),
                legend=dict(title="Loan Status"),
                margin=dict(t=20, b=42, l=45, r=10)
            )
            st.plotly_chart(ch_fig, width="stretch", config={"displayModeBar": False})

    # Property Area
    with row1c2:
        with st.container(border=True):
            st.markdown(
                '<div class="chart-title">Approval Rate by Property Area</div>'
                '<div class="chart-subtitle">Approval patterns across property areas.</div>',
                unsafe_allow_html=True
            )

            pa = pd.crosstab(
                raw["Property_Area"], raw["Loan_Status"], normalize="index"
            ).get("Y", pd.Series(dtype=float)) * 100

            property_order = ["Rural", "Semiurban", "Urban"]
            pa_values = [pa.get(area, 0) for area in property_order]

            pa_fig = go.Figure(go.Bar(
                x=property_order,
                y=pa_values,
                marker=dict(color=[ACCENT_COLOR if v >= 68.7 else NOT_APPROVED for v in pa_values]),
                text=[f"{v:.1f}%" for v in pa_values],
                textposition="outside",
                cliponaxis=False,
                width=0.52,
            ))
            pa_fig.update_traces(marker_cornerradius=6, cliponaxis=False)
            pa_fig = style_fig(pa_fig, height=245)
            pa_fig.update_layout(
                yaxis=dict(title="Approval Rate (%)", range=[0, 115], gridcolor=GRID_COLOR),
                xaxis=dict(gridcolor=GRID_COLOR),
                showlegend=False,
                margin=dict(t=20, b=42, l=45, r=10),
            )
            pa_fig.add_hline(
                y=68.7, line_dash="dash", line_color=TEXT_MUTED,
                annotation_text="Avg 68.7%", annotation_position="top left"
            )
            st.plotly_chart(pa_fig, width="stretch", config={"displayModeBar": False})

    # ---------------- ROW 2 ----------------
    row2c1, row2c2 = st.columns(2)

    # Education
    with row2c1:
        with st.container(border=True):
            st.markdown(
                '<div class="chart-title">Approval Rate by Education</div>'
                '<div class="chart-subtitle">Approval rate by education level.</div>',
                unsafe_allow_html=True
            )

            edu = pd.crosstab(
                raw["Education"], raw["Loan_Status"], normalize="index"
            ).get("Y", pd.Series(dtype=float)) * 100

            edu_order = ["Not Graduate", "Graduate"]
            edu_values = [edu.get(level, 0) for level in edu_order]

            edu_fig = go.Figure(go.Bar(
                x=edu_order,
                y=edu_values,
                marker=dict(color=[NOT_APPROVED if v < 68.7 else ACCENT_COLOR for v in edu_values]),
                text=[f"{v:.1f}%" for v in edu_values],
                textposition="outside",
                cliponaxis=False,
                width=0.45,
            ))
            edu_fig.update_traces(marker_cornerradius=6, cliponaxis=False)
            edu_fig = style_fig(edu_fig, height=235)
            edu_fig.update_layout(
                yaxis=dict(title="Approval Rate (%)", range=[0, 115], gridcolor=GRID_COLOR),
                xaxis=dict(gridcolor=GRID_COLOR),
                showlegend=False,
                margin=dict(t=20, b=42, l=45, r=10),
            )
            edu_fig.add_hline(
                y=68.7, line_dash="dash", line_color=TEXT_MUTED,
                annotation_text="Avg 68.7%", annotation_position="top left"
            )
            st.plotly_chart(edu_fig, width="stretch", config={"displayModeBar": False})

    # Self Employed
    with row2c2:
        with st.container(border=True):
            st.markdown(
                '<div class="chart-title">Approval Rate by Self Employed</div>'
                '<div class="chart-subtitle">Approval rate for self-employed vs salaried.</div>',
                unsafe_allow_html=True
            )

            se = pd.crosstab(
                raw["Self_Employed"], raw["Loan_Status"], normalize="index"
            ).get("Y", pd.Series(dtype=float)) * 100

            se_order = ["No", "Yes"]
            se_values = [se.get(status, 0) for status in se_order]

            se_fig = go.Figure(go.Bar(
                x=["Salaried (No)", "Self Employed (Yes)"],
                y=se_values,
                marker=dict(color=[NOT_APPROVED if v < 68.7 else ACCENT_COLOR for v in se_values]),
                text=[f"{v:.1f}%" for v in se_values],
                textposition="outside",
                cliponaxis=False,
                width=0.45,
            ))
            se_fig.update_traces(marker_cornerradius=6, cliponaxis=False)
            se_fig = style_fig(se_fig, height=235)
            se_fig.update_layout(
                yaxis=dict(title="Approval Rate (%)", range=[0, 115], gridcolor=GRID_COLOR),
                xaxis=dict(gridcolor=GRID_COLOR),
                showlegend=False,
                margin=dict(t=20, b=42, l=45, r=10),
            )
            se_fig.add_hline(
                y=68.7, line_dash="dash", line_color=TEXT_MUTED,
                annotation_text="Avg 68.7%", annotation_position="top left"
            )
            st.plotly_chart(se_fig, width="stretch", config={"displayModeBar": False})

    # ---------------- SCATTER ----------------
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Applicant Income vs Loan Amount</div>'
            '<div class="chart-subtitle">Relationship between income and requested loan amount.</div>',
            unsafe_allow_html=True
        )

        status_colors = raw["Loan_Status"].map({"Y": APPROVED, "N": NOT_APPROVED})

        scatter_fig = go.Figure(go.Scatter(
            x=raw["ApplicantIncome"],
            y=raw["LoanAmount"],
            mode="markers",
            marker=dict(color=status_colors, size=6, opacity=0.7),
            text=raw["Loan_Status"],
            showlegend=False,
            hovertemplate=(
                "Applicant Income: ₹%{x:,.0f}<br>"
                "Loan Amount: ₹%{y:,.0f}k<br>"
                "Loan Status: %{text}<extra></extra>"
            )
        ))

        for status, color in [("Approved", APPROVED), ("Not Approved", NOT_APPROVED)]:
            scatter_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=color, size=8), name=status
            ))

        scatter_fig = style_fig(scatter_fig, height=245)
        scatter_fig.update_layout(
            xaxis=dict(title="Applicant Income (₹)", gridcolor=GRID_COLOR),
            yaxis=dict(title="Loan Amount (₹k)", gridcolor=GRID_COLOR),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(t=28, b=42, l=55, r=10),
        )
        st.plotly_chart(scatter_fig, width="stretch", config={"displayModeBar": False})

    # ---------------- CORRELATION HEATMAP ----------------
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Correlation Between Numeric Features</div>'
            '<div class="chart-subtitle">Relationship between the main numeric variables.</div>',
            unsafe_allow_html=True
        )

        numeric_cols = [
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History"
        ]

        corr = raw[numeric_cols].corr()

        heat_fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=[[0, NOT_APPROVED], [0.5, "#F8F9FC"], [1, ACCENT_COLOR]],
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        ))

        heat_fig = style_fig(heat_fig, height=330)
        heat_fig.update_layout(
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR),
            margin=dict(t=10, b=60, l=105, r=20),
        )
        st.plotly_chart(heat_fig, width="stretch", config={"displayModeBar": False})

# ==================================================================
# PAGE 4: LOAN FORM
# ==================================================================
elif st.session_state.page == "Loan_Form":

    render_page_header(
        "Loan Application Form",
        "Individual applicant assessment &nbsp;&bull;&nbsp; Instant eligibility prediction",
        [
            ("Single Applicant", "rgba(146, 130, 255, 0.12)", ACCENT_COLOR, BORDER),
            ("Random Forest", "rgba(146, 130, 255, 0.12)", APPROVED, BORDER),
            ("Instant Decision", "rgba(224, 168, 56, 0.15)", WARNING, BORDER),
        ]
    )

    with st.form("loan_form"):

        with st.container(border=True):
            st.subheader("Personal Information")
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"], index=None, placeholder="Select Gender")
            with col2:
                dependents = st.selectbox("Dependents", [0, 1, 2, 3], index=None, placeholder="Select Number of Dependents")

            col1, col2 = st.columns(2)
            with col1:
                education = st.selectbox("Education", ["Not Graduate", "Graduate"], index=None, placeholder="Select Education Level")
            with col2:
                self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=None, placeholder="Select Employment Status")

        with st.container(border=True):
            st.subheader("Financial Information")
            col1, col2 = st.columns(2)
            with col1:
                applicant_income = st.number_input("Applicant Income (₹)", min_value=0.0, step=1000.0, format="%.0f")
            with col2:
                coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0.0, step=1000.0, format="%.0f")

            col1, col2 = st.columns(2)
            with col1:
                loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0, step=1000.0, format="%.0f")
            with col2:
                loan_amount_term = st.selectbox(
                    "Loan Term", [120, 180, 240, 300, 360],
                    format_func=lambda m: f"{m} months ({m // 12} years)",
                )

        with st.container(border=True):
            st.subheader("Credit & Property Information")
            col1, col2 = st.columns(2)
            with col1:
                credit_history = st.selectbox("Previous loan/credit bills paid on time?", ["Yes", "No"], index=None, placeholder="Select Credit History")
            with col2:
                property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"], index=None, placeholder="Select Property Area")

        submit = st.form_submit_button("Check Loan Eligibility", width="stretch")

    # ---------------- PREDICTION ----------------
    if submit:

        gender_value = 1 if gender == "Male" else 0
        education_value = 1 if education == "Graduate" else 0
        self_employed_value = 1 if self_employed == "Yes" else 0
        credit_history_value = 1.0 if credit_history == "Yes" else 0.0

        if property_area == "Semiurban":
            semiurban, urban = 1, 0
        elif property_area == "Urban":
            semiurban, urban = 0, 1
        else:
            semiurban, urban = 0, 0

        total_income = applicant_income + coapplicant_income

        input_data = pd.DataFrame([[
            gender_value, dependents, education_value, self_employed_value,
            applicant_income, coapplicant_income, loan_amount, loan_amount_term,
            credit_history_value, semiurban, urban, total_income
        ]], columns=FEATURE_NAMES)

        # Random Forest was trained on RAW (unscaled) features — predict directly.
        rf_prediction = rf_model.predict(input_data)
        rf_proba = rf_model.predict_proba(input_data)[0]  # [P(Not Approved), P(Approved)]
        confidence = round(max(rf_proba) * 100, 1)

        rf_result = "Approved" if rf_prediction[0] == 1 else "Not Approved"
        result_color = APPROVED if rf_result == "Approved" else NOT_APPROVED

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "result": rf_result,
            "confidence": confidence,
            "total_income": total_income,
            "loan_amount": loan_amount,
        })

        top_col1, top_col2 = st.columns([1, 1])

        with top_col1:
            with st.container(border=True):
                if rf_result == "Approved":
                    st.success("Loan Approved")
                    st.write("The model predicts that your loan application is eligible.")
                else:
                    st.error("Loan Not Approved")
                    st.write("The model predicts that your loan application is not eligible.")
                st.markdown(f"""
                    <div class="progress-row" style="margin-top:14px;">
                        <div class="progress-label"><span>Model confidence</span><span>{confidence:.1f}%</span></div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width:{confidence}%; background:{result_color};"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with top_col2:
            with st.container(border=True):
                st.markdown("##### Prediction Confidence")
                conf_fig = go.Figure()
                conf_fig.add_trace(go.Bar(
                    x=[rf_proba[1] * 100], y=["Approved"], orientation="h",
                    marker=dict(color=APPROVED, cornerradius=6),
                    text=[f"{rf_proba[1]*100:.1f}%"], textposition="outside", cliponaxis=False,
                ))
                conf_fig.add_trace(go.Bar(
                    x=[rf_proba[0] * 100], y=["Not Approved"], orientation="h",
                    marker=dict(color=NOT_APPROVED, cornerradius=6),
                    text=[f"{rf_proba[0]*100:.1f}%"], textposition="outside", cliponaxis=False,
                ))
                conf_fig.update_traces(cliponaxis=False)
                conf_fig = style_fig(conf_fig, height=220)
                conf_fig.update_layout(
                    xaxis=dict(range=[0, 115], title="Probability (%)", gridcolor=GRID_COLOR),
                    yaxis=dict(title=""),
                    showlegend=False,
                    margin=dict(t=40, b=30, l=10, r=40),
                )
                st.plotly_chart(conf_fig, width="stretch")
                st.caption(f"Model is {confidence:.1f}% confident, driven mainly by Credit History and Total Income.")


# ==================================================================
# PAGE 5: BULK PREDICTION (SUPPORTS CSV, EXCEL, AND JSON)
# ==================================================================
elif st.session_state.page == "Bulk_Prediction":

    import difflib
    import json
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)

    render_page_header(
        "Bulk Prediction",
        "Batch processing &nbsp;&bull;&nbsp; Upload CSV, Excel or JSON for automated decisions",
        [
            ("Batch Mode", "rgba(146, 130, 255, 0.12)", ACCENT_COLOR, BORDER),
            ("CSV / Excel / JSON", "rgba(146, 130, 255, 0.12)", APPROVED, BORDER),
            ("Strict Validation", "rgba(224, 168, 56, 0.15)", WARNING, BORDER),
        ]
    )

    # --------------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------------
    REQUIRED_RAW_COLUMNS = [
        "Gender", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area",
    ]

    # A row with an empty/invalid value here is skipped (never guessed).
    CRITICAL_COLUMNS = ["Credit_History", "ApplicantIncome", "LoanAmount"]
    NONCRITICAL_COLUMNS = [c for c in REQUIRED_RAW_COLUMNS if c not in CRITICAL_COLUMNS]

    CATEGORICAL_COLUMNS = ["Gender", "Education", "Self_Employed", "Property_Area"]
    NUMERIC_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

    NOT_IN_FILE = "-- Not in my file --"
    NO_LABEL = "-- None --"

    COLUMN_ALIASES = {
        "Gender": ["gender", "sex"],
        "Dependents": ["dependents", "dependent", "no_of_dependents",
                       "number_of_dependents", "num_dependents", "family_dependents"],
        "Education": ["education", "education_level", "edu", "degree", "qualification"],
        "Self_Employed": ["self_employed", "selfemployed", "employment",
                          "employment_status", "job_status"],
        "ApplicantIncome": ["applicantincome", "applicant_income", "income",
                            "salary", "monthly_income", "annual_income"],
        "CoapplicantIncome": ["coapplicantincome", "coapplicant_income", "co_income",
                              "partner_income", "spouse_income"],
        "LoanAmount": ["loanamount", "loan_amount", "loan_amt", "requested_loan",
                       "requested_amount"],
        "Loan_Amount_Term": ["loan_amount_term", "loan_term", "term",
                             "loan_term_months", "repayment_months", "tenure"],
        "Credit_History": ["credit_history", "credit_record", "credit_score",
                           "cibil", "cibil_score"],
        "Property_Area": ["property_area", "property", "area", "location",
                          "property_location"],
    }

    # Accepted spellings -> model label
    BASE_MAPS = {
        "Gender": {"male": "Male", "m": "Male", "female": "Female", "f": "Female"},
        "Education": {"graduate": "Graduate", "not graduate": "Not Graduate",
                      "not_graduate": "Not Graduate", "undergraduate": "Not Graduate"},
        "Self_Employed": {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "1.0": "Yes",
                          "self employed": "Yes", "self-employed": "Yes",
                          "self_employed": "Yes", "selfemployed": "Yes",
                          "no": "No", "n": "No", "false": "No", "0": "No", "0.0": "No",
                          "employed": "No", "salaried": "No", "not self employed": "No",
                          "not self-employed": "No", "unemployed": "No"},
        "Property_Area": {"urban": "Urban", "semiurban": "Semiurban",
                          "semi urban": "Semiurban", "semi-urban": "Semiurban",
                          "rural": "Rural"},
    }

    TARGET_OPTIONS = {
        "Gender": ["Male", "Female"],
        "Education": ["Graduate", "Not Graduate"],
        "Self_Employed": ["No", "Yes"],
        "Property_Area": ["Rural", "Semiurban", "Urban"],
    }

    # Only used if the training file cannot be read.
    FALLBACK_DEFAULTS = {
        "Gender": "Male", "Dependents": 0, "Education": "Graduate",
        "Self_Employed": "No", "CoapplicantIncome": 1200.0,
        "Loan_Amount_Term": 360.0, "Property_Area": "Semiurban",
    }

    SCALE_OPTIONS = {
        "No change": 1.0,
        "÷ 12  (annual → monthly)": 1 / 12,
        "÷ 1000  (rupees → thousands)": 1 / 1000,
        "× 12  (years → months)": 12.0,
        "× 1000  (thousands → rupees)": 1000.0,
    }

    # Words that mean Approved / Not approved in an "actual outcome" column.
    POSITIVE_WORDS = {"y", "yes", "1", "1.0", "true", "t", "approved", "approve"}
    NEGATIVE_WORDS = {"n", "no", "0", "0.0", "false", "f", "rejected", "not approved",
                      "denied", "declined"}
    LABEL_HINTS = {"loanstatus", "loanapproved", "approved", "status", "target",
                   "label", "outcome", "approval", "actual"}

    # --------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------
    def normalize_column_name(name):
        return "".join(ch for ch in str(name).lower().strip() if ch.isalnum())

    def find_column_mapping(uploaded_columns):
        norm_to_col = {normalize_column_name(c): c for c in uploaded_columns}
        mapping, used = {}, set()

        # 1) exact / alias match
        for req in REQUIRED_RAW_COLUMNS:
            for name in [req] + COLUMN_ALIASES.get(req, []):
                cand = norm_to_col.get(normalize_column_name(name))
                if cand is not None and cand not in used:
                    mapping[req] = cand
                    used.add(cand)
                    break

        # 2) fuzzy match for whatever is left
        for req in REQUIRED_RAW_COLUMNS:
            if req in mapping:
                continue
            free = {n: c for n, c in norm_to_col.items() if c not in used}
            hit = difflib.get_close_matches(normalize_column_name(req), list(free),
                                            n=1, cutoff=0.85)
            if hit:
                mapping[req] = free[hit[0]]
                used.add(free[hit[0]])
        return mapping

    def parse_dependents(series):
        cleaned = series.astype(str).str.replace("+", "", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")

    @st.cache_data
    def get_training_defaults():
        """Values used ONLY for empty cells in non-critical columns.
        Median for numbers, most common value for categories - from TRAINING data."""
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
        """Typical size of each numeric column in the TRAINING data (for the unit check)."""
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
        """Pick the unit fix that brings the file closest to the training scale."""
        best_label, best_err = "No change", abs(np.log(ratio))
        for label, factor in SCALE_OPTIONS.items():
            err = abs(np.log(ratio * factor))
            if err < best_err - 1e-9:
                best_label, best_err = label, err
        return best_label

    def blocked(message):
        """Show why prediction is blocked + a disabled button, then stop the page."""
        st.error(message)
        st.button("Run Bulk Prediction", type="primary", width="stretch",
                  disabled=True, key="bulk_predict_disabled")
        st.caption("Fix the issue above to enable prediction.")
        st.stop()

    training_defaults = get_training_defaults()

    # --------------------------------------------------------------
    # UPLOAD + TEMPLATE (CSV, XLSX, JSON)
    # --------------------------------------------------------------
    st.subheader("Upload Applicant Data")

    up_col, tpl_col = st.columns([3, 1])
    with up_col:
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel or JSON file", type=["csv", "xlsx", "json"],
            help="Upload applicant data in CSV, Excel or JSON format.",
        )
    with tpl_col:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        sample_data = pd.DataFrame({
            "Gender": ["Male", "Female"], "Dependents": ["0", "1"],
            "Education": ["Graduate", "Graduate"], "Self_Employed": ["No", "Yes"],
            "ApplicantIncome": [5000, 3500], "CoapplicantIncome": [1500, 1000],
            "LoanAmount": [150, 120], "Loan_Amount_Term": [360, 360],
            "Credit_History": [1, 1], "Property_Area": ["Urban", "Semiurban"],
        })
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.download_button("Sample CSV", data=sample_data.to_csv(index=False),
                               file_name="loan_prediction_template.csv", mime="text/csv",
                               width="stretch")
        with d_c2:
            st.download_button("Sample JSON", data=sample_data.to_json(orient="records", indent=2),
                               file_name="loan_prediction_template.json", mime="application/json",
                               width="stretch")

    if uploaded_file is None:
        st.info("Upload a CSV, Excel or JSON file to get started. Column names don't need to match exactly.")
        st.stop()

    # --------------------------------------------------------------
    # READ FILE (SUPPORTS CSV, EXCEL, JSON)
    # --------------------------------------------------------------
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file, encoding="utf-8")
        elif fname.endswith(".json"):
            import json as py_json
            import io
            content = uploaded_file.read()
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin1")
            parsed_json = py_json.loads(text_content)
            if isinstance(parsed_json, list):
                raw_df = pd.DataFrame(parsed_json)
            elif isinstance(parsed_json, dict):
                found_key = False
                for k in ["data", "records", "applicants", "rows", "items"]:
                    if k in parsed_json and isinstance(parsed_json[k], list):
                        raw_df = pd.DataFrame(parsed_json[k])
                        found_key = True
                        break
                if not found_key:
                    raw_df = pd.DataFrame(parsed_json)
            else:
                raw_df = pd.read_json(io.StringIO(text_content))
        else:
            raw_df = pd.read_excel(uploaded_file)
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding="latin1")
        except Exception as e:
            st.error(f"Unable to read the uploaded file: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Unable to read the uploaded file: {e}")
        st.stop()

    if raw_df.empty:
        st.error("The uploaded file is empty.")
        st.stop()

    raw_df = raw_df.reset_index(drop=True)
    uploaded_columns = raw_df.columns.tolist()
    fid = f"{uploaded_file.name}_{uploaded_file.size}"   # keeps widget state per file

    st.success(f"File uploaded successfully — {len(raw_df)} applicants found.")
    with st.expander("View Uploaded Data", expanded=False):
        st.dataframe(raw_df, width="stretch", height=300)

    # --------------------------------------------------------------
    # COLUMN MAPPING
    # --------------------------------------------------------------
    st.subheader("Column Mapping")
    summary_box = st.container()          # filled after the final mapping is known

    auto_mapping = find_column_mapping(uploaded_columns)
    mapping = dict(auto_mapping)
    unmatched = [c for c in REQUIRED_RAW_COLUMNS if c not in auto_mapping]

    with st.expander("Review / fix column mapping", expanded=bool(unmatched)):
        if unmatched:
            st.caption("We couldn't match these fields automatically. If your file has "
                       "the data under another name, pick that column.")
            grid = st.columns(2)
            for i, col in enumerate(unmatched):
                with grid[i % 2]:
                    choice = st.selectbox(col, [NOT_IN_FILE] + uploaded_columns,
                                          key=f"bulk_map_{fid}_{col}")
                    if choice != NOT_IN_FILE:
                        mapping[col] = choice

        table_rows = []
        for col in REQUIRED_RAW_COLUMNS:
            if col in auto_mapping:
                status = "Matched (auto)"
            elif col in mapping:
                status = "Matched (manual)"
            else:
                status = "Missing"
            table_rows.append({"Model field": col,
                               "Mapped column": mapping.get(col, "—"),
                               "Status": status})
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
        blocked("The same uploaded column is mapped to more than one field: "
                + ", ".join(dupes))

    # ---- STRICT GATE: any missing column blocks prediction ----
    if missing:
        blocked(
            f"Prediction blocked: {len(missing)} required column(s) not found — "
            f"{', '.join(missing)}. The model needs all {len(REQUIRED_RAW_COLUMNS)} inputs, "
            "and filling a whole column with guessed values gives wrong predictions. "
            "If your file has the data under another name, pick it in the mapping above. "
            "Otherwise add the column and upload again (see the sample template)."
        )

    # --------------------------------------------------------------
    # CREDIT SCORE -> CREDIT_HISTORY (0/1)
    # --------------------------------------------------------------
    credit_threshold = None
    ch_num = pd.to_numeric(raw_df[mapping["Credit_History"]], errors="coerce")
    if ch_num.notna().mean() > 0.5 and ch_num.max() > 1:
        lo, hi = int(ch_num.min()), int(ch_num.max())
        with st.expander("Credit score conversion", expanded=True):
            st.info(
                f"`{mapping['Credit_History']}` looks like a credit score ({lo}—{hi}), "
                "but the model needs Credit_History as 1 (good) or 0 (bad). "
                "Choose the cut-off:"
            )
            if lo < hi:
                credit_threshold = st.slider(
                    "Scores at or above this value count as good credit (1)",
                    lo, hi, int(min(max(650, lo), hi)), key=f"bulk_credit_{fid}",
                )
            else:
                credit_threshold = lo

    # --------------------------------------------------------------
    # BUILD STANDARDISED DATAFRAME
    # --------------------------------------------------------------
    df = pd.DataFrame(index=raw_df.index)
    for col in REQUIRED_RAW_COLUMNS:
        df[col] = raw_df[mapping[col]].copy()

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col].astype(str).str.strip()
                       .replace({"nan": np.nan, "None": np.nan, "NaN": np.nan, "": np.nan}))

    # --------------------------------------------------------------
    # CATEGORICAL VALUES
    # --------------------------------------------------------------
    unresolved = []
    value_maps_used = {}
    for col in CATEGORICAL_COLUMNS:
        cleaned = df[col].astype(str).str.strip().str.lower()
        was_na = df[col].isna()
        mapped = cleaned.map(BASE_MAPS[col])
        unknown_vals = sorted(cleaned[mapped.isna() & ~was_na].unique())

        if unknown_vals:
            user_map = {}
            with st.expander(f"Fix values in {col}", expanded=True):
                st.caption(f"These values in `{mapping[col]}` weren't recognised. "
                           "Tell us what each one means:")
                grid = st.columns(2)
                for i, val in enumerate(unknown_vals):
                    with grid[i % 2]:
                        pick = st.selectbox(
                            f"`{val}` means", TARGET_OPTIONS[col], index=None,
                            placeholder="Select...", key=f"bulk_val_{fid}_{col}_{val}",
                        )
                        if pick is not None:
                            user_map[val] = pick
            value_maps_used[col] = user_map
            mapped = cleaned.map({**BASE_MAPS[col], **user_map})
            if any(v not in user_map for v in unknown_vals):
                unresolved.append(col)

        mapped[was_na] = np.nan
        df[col] = mapped

    if unresolved:
        blocked("Please tell us what the unrecognised values mean in: "
                + ", ".join(unresolved))

    # --------------------------------------------------------------
    # NUMERIC VALUES
    # --------------------------------------------------------------
    df["Dependents"] = parse_dependents(df["Dependents"])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    all_empty = [c for c in df.columns if df[c].isna().all()]
    if all_empty:
        blocked("These columns have no usable values (empty or not in the expected "
                f"format): {', '.join(all_empty)}.")

    num = pd.to_numeric(df["Credit_History"], errors="coerce")
    if credit_threshold is not None:
        df["Credit_History"] = (num >= credit_threshold).astype(float).where(num.notna())
    else:
        text_ch = df["Credit_History"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0})
        df["Credit_History"] = num.where(num.isin([0, 1])).fillna(text_ch)

    # --------------------------------------------------------------
    # UNIT CHECK
    # --------------------------------------------------------------
    train_med = get_training_medians()
    scale_choice = {}
    flagged = {}
    for col in NUMERIC_COLUMNS:
        pos = df[col][df[col] > 0]
        t = train_med.get(col)
        if len(pos) and t:
            ratio = float(pos.median()) / t
            if ratio > 5 or ratio < 0.2:
                flagged[col] = ratio

    if flagged:
        with st.expander("Unit check", expanded=True):
            st.warning("Some columns look like they use different units than the data the "
                       "model was trained on. A suggested fix is pre-selected. "
                       "Change it if it is wrong.")
            labels = list(SCALE_OPTIONS)
            for col, ratio in flagged.items():
                choice = st.selectbox(
                    f"{col}  (typical value in your file is {ratio:.1f}× the training data)",
                    labels, index=labels.index(best_scale(ratio)),
                    key=f"bulk_scale_{fid}_{col}",
                )
                scale_choice[col] = choice
                df[col] = df[col] * SCALE_OPTIONS[choice]

    # --------------------------------------------------------------
    # ROW VALIDITY + FILLING EMPTY CELLS
    # --------------------------------------------------------------
    valid_mask = ~df[CRITICAL_COLUMNS].isna().any(axis=1)
    imputed_per_row = df[NONCRITICAL_COLUMNS].isna().sum(axis=1)

    for col in NONCRITICAL_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(training_defaults[col])

    n_total = len(df)
    n_valid = int(valid_mask.sum())
    n_skipped = n_total - n_valid

    if n_valid == 0:
        blocked("No applicant has all of Credit_History, ApplicantIncome and LoanAmount "
                "filled in with valid values, so nothing can be predicted.")

    # --------------------------------------------------------------
    # FEATURE ENGINEERING + ENCODING
    # --------------------------------------------------------------
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

    # Reliability per row: how many (non-critical) cells had to be filled from training data
    rel_valid = np.where(imputed_per_row[valid_mask] == 0, "High",
                 np.where(imputed_per_row[valid_mask] <= 2, "Medium", "Low"))

    st.success(f"Data preprocessing completed. {n_valid} of {n_total} applicants "
               "are ready for prediction.")
    if n_skipped:
        st.warning(f"{n_skipped} applicant(s) have an empty or invalid Credit_History, "
                   "ApplicantIncome or LoanAmount. They will be skipped (shown as "
                   "'Not predicted') instead of being guessed.")
    n_filled = int((imputed_per_row[valid_mask] > 0).sum())
    if n_filled:
        st.info(f"{n_filled} applicant(s) had some empty optional cells. Those were filled "
                "with typical values from the training data and marked Medium/Low in the "
                "'Reliability' column.")

    with st.expander("View Processed Model Features", expanded=False):
        st.dataframe(X_valid, width="stretch", height=300)

    # --------------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------------
    settings_sig = json.dumps(
        [fid, sorted(mapping.items()), credit_threshold,
         sorted(scale_choice.items()), value_maps_used],
        sort_keys=True, default=str,
    )

    st.markdown("---")
    predict_bulk = st.button("Run Bulk Prediction", type="primary", width="stretch",
                             key="bulk_predict_btn")

    if predict_bulk:
        n = len(X_valid)
        chunk = 5000
        preds, probs = [], []
        bar = st.progress(0.0, text="Predicting...")
        for start in range(0, n, chunk):
            part = X_valid.iloc[start:start + chunk]
            preds.append(rf_model.predict(part))
            probs.append(rf_model.predict_proba(part))
            bar.progress(min((start + chunk) / n, 1.0), text="Predicting...")
        bar.empty()

        predictions = np.concatenate(preds)
        probabilities = np.vstack(probs)

        results = raw_df.copy()
        results["Prediction"] = "Not predicted"
        results["Confidence (%)"] = np.nan
        results["Reliability"] = "N/A"
        results.loc[X_valid.index, "Prediction"] = np.where(predictions == 1,
                                                            "Approved", "Not Approved")
        results.loc[X_valid.index, "Confidence (%)"] = (probabilities.max(axis=1) * 100).round(1)
        results.loc[X_valid.index, "Reliability"] = rel_valid

        st.session_state["bulk_results"] = {"sig": settings_sig, "df": results}

    # --------------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------------
    saved = st.session_state.get("bulk_results")

    if saved and saved["sig"] != settings_sig:
        st.info("The file or settings changed since the last run. "
                "Click **Run Bulk Prediction** again to refresh the results.")

    if saved and saved["sig"] == settings_sig:

        results_df = saved["df"]
        done = results_df[results_df["Prediction"] != "Not predicted"]
        total = len(results_df)
        predicted_n = len(done)
        approved_n = int((done["Prediction"] == "Approved").sum())
        rejected_n = predicted_n - approved_n
        approval_rate = approved_n / predicted_n * 100 if predicted_n else 0
        avg_conf = done["Confidence (%)"].mean() if predicted_n else 0

        st.subheader("Prediction Summary")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Applicants", total)
        m2.metric("Predicted", predicted_n)
        m3.metric("Approved", approved_n)
        m4.metric("Not Approved", rejected_n)
        m5.metric("Approval Rate", f"{approval_rate:.1f}%")
        m6.metric("Avg Confidence", f"{avg_conf:.1f}%")

        ch1, ch2 = st.columns(2)
        with ch1:
            with st.container(border=True):
                st.markdown("**Approved vs Not Approved**")
                fig = go.Figure(go.Bar(
                    x=["Approved", "Not Approved"], y=[approved_n, rejected_n],
                    marker=dict(color=[APPROVED, NOT_APPROVED]),
                    text=[approved_n, rejected_n], textposition="outside",
                    width=0.5, cliponaxis=False,
                ))
                fig.update_traces(marker_cornerradius=6, cliponaxis=False)
                fig = style_fig(fig, height=260)
                fig.update_layout(
                    showlegend=False,
                    yaxis=dict(title="Applicants", gridcolor=GRID_COLOR,
                               range=[0, max(approved_n, rejected_n, 1) * 1.25]),
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with ch2:
            with st.container(border=True):
                st.markdown("**Confidence distribution**")
                fig = go.Figure(go.Histogram(
                    x=done["Confidence (%)"], nbinsx=20,
                    marker=dict(color=ACCENT_COLOR),
                ))
                fig = style_fig(fig, height=260)
                fig.update_layout(showlegend=False,
                                  xaxis=dict(title="Confidence (%)", gridcolor=GRID_COLOR),
                                  yaxis=dict(title="Applicants", gridcolor=GRID_COLOR))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # ---- filters ----
        st.subheader("Prediction Results")
        f1, f2, f3 = st.columns(3)
        with f1:
            pred_filter = st.multiselect(
                "Prediction", ["Approved", "Not Approved", "Not predicted"],
                default=["Approved", "Not Approved", "Not predicted"])
        with f2:
            rel_filter = st.multiselect(
                "Reliability", ["High", "Medium", "Low", "N/A"],
                default=["High", "Medium", "Low", "N/A"])
        with f3:
            min_conf = st.slider("Minimum confidence (%)", 0, 100, 0)

        view = results_df[
            results_df["Prediction"].isin(pred_filter)
            & results_df["Reliability"].isin(rel_filter)
            & ((results_df["Confidence (%)"] >= min_conf) | results_df["Confidence (%)"].isna())
        ].sort_values("Confidence (%)", ascending=False, na_position="last")

        st.caption(f"Showing {len(view)} of {total} applicants.")
        st.dataframe(view, width="stretch", height=450)

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.download_button(
                "Download Prediction Results (CSV)",
                data=view.to_csv(index=False),
                file_name="loan_bulk_prediction_results.csv",
                mime="text/csv",
                width="stretch",
            )
        with res_c2:
            st.download_button(
                "Download Prediction Results (JSON)",
                data=view.to_json(orient="records", indent=2),
                file_name="loan_bulk_prediction_results.json",
                mime="application/json",
                width="stretch",
            )

        # ---- optional: compare with the real outcome column ----
        with st.expander("Check accuracy against the actual outcome (optional)", expanded=False):
            st.caption("If your file already has the real result (approved / not approved), "
                       "pick that column to see how well the model did.")
            original_cols = list(raw_df.columns)
            guess = next((c for c in original_cols
                          if normalize_column_name(c) in LABEL_HINTS), None)
            options = [NO_LABEL] + original_cols
            label_col = st.selectbox(
                "Actual outcome column", options,
                index=options.index(guess) if guess else 0,
                key=f"bulk_label_{fid}",
            )

            if label_col != NO_LABEL:
                actual_text = results_df[label_col].astype(str).str.strip().str.lower()
                actual = actual_text.map(
                    lambda v: 1 if v in POSITIVE_WORDS else (0 if v in NEGATIVE_WORDS else np.nan))
                predicted = results_df["Prediction"].map({"Approved": 1, "Not Approved": 0})
                ok = actual.notna() & predicted.notna()

                if ok.sum() == 0:
                    st.warning(f"Could not read `{label_col}` as approved / not approved "
                               "values, so accuracy can't be calculated.")
                else:
                    y_true = actual[ok].astype(int)
                    y_pred = predicted[ok].astype(int)
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Accuracy", f"{accuracy_score(y_true, y_pred) * 100:.1f}%")
                    a2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0) * 100:.1f}%")
                    a3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0) * 100:.1f}%")
                    a4.metric("F1 score", f"{f1_score(y_true, y_pred, zero_division=0) * 100:.1f}%")

                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    st.dataframe(
                        pd.DataFrame(cm,
                                     index=["Actual: Not approved", "Actual: Approved"],
                                     columns=["Predicted: Not approved", "Predicted: Approved"]),
                        width="stretch",
                    )
                    st.caption(f"Based on {int(ok.sum())} applicants that have both a "
                               "prediction and a readable actual outcome.")
