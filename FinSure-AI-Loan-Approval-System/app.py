# ============================================================
# FINSURE ANALYTICS
# EXECUTIVE DASHBOARD
# PHASE 11.1
# ============================================================

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Loan Intelligence Platform",

    layout="wide",

    initial_sidebar_state="expanded",

)

# ============================================================
# LOAD CSS
# ============================================================

css_file = (
    Path(__file__).parent
    / "assets"
    / "styles.css"
)

if css_file.exists():

    with open(
        css_file,
        encoding="utf-8",
    ) as file:

        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True,
        )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="
        color:#5D4037;
        margin-bottom:0px;
        ">
        Loan Intelligence
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="
        color:#4E342E;
        font-size:16px;
        font-weight:500;
        ">
        Decision Support Platform
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div style="
        color:#3E2723;
        font-size:18px;
        line-height:1.8;
        ">
        Professional platform for
        credit evaluation,
        approval assessment,
        and lending analytics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.success(
        "Prediction Engine Operational"
    )

    st.markdown(
        """
        <p style="
        color:#4E342E;
        font-size:15px;
        font-weight:500;
        ">
        Production Build
        </p>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================

st.title(
    "Loan Decision Intelligence Platform"
)

st.markdown(
    """
    Professional credit risk assessment,
    approval prediction,
    and lending analytics solution.
    """
)

# ============================================================
# KPI SECTION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Model Status",
        "Active"
    )

with col2:

    st.metric(
        "Approval Engine",
        "Operational"
    )

with col3:

    st.metric(
        "Risk Framework",
        "Enabled"
    )

with col4:

    st.metric(
        "Build",
        "Production"
    )

# ============================================================
# PLATFORM OVERVIEW
# ============================================================

st.markdown(
    """
    <div class='fin-card'>

    <h3>Platform Overview</h3>

    <p>
    The platform evaluates applicant eligibility,
    predicts approval probability,
    assigns risk classifications,
    and generates decision-support insights
    for lending operations.
    </p>

    <p>
    The predictive framework has been trained
    on loan application data and supports
    consistent lending decisions through
    quantitative risk evaluation.
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PROJECT STATISTICS
# ============================================================

st.subheader(
    "Project Statistics"
)

s1, s2, s3, s4 = st.columns(4)

with s1:

    st.metric(
        "Dataset Records",
        "2,000"
    )

with s2:

    st.metric(
        "Features",
        "6"
    )

with s3:

    st.metric(
        "Accuracy",
        "100%"
    )

with s4:

    st.metric(
        "ROC AUC",
        "1.00"
    )

# ============================================================
# VISUAL ANALYTICS
# ============================================================

left, right = st.columns(2)

# ------------------------------------------------------------
# LOAN APPROVAL DISTRIBUTION
# ------------------------------------------------------------

with left:

    st.subheader(
        "Loan Approval Distribution"
    )

    pie_df = pd.DataFrame({

        "Status": [
            "Approved",
            "Rejected",
        ],

        "Count": [
            342,
            1658,
        ],

    })

    fig = px.pie(

        pie_df,

        names="Status",

        values="Count",

        color="Status",

        color_discrete_map={

            "Approved": "#6D9773",

            "Rejected": "#C07A2C",

        },

    )

    fig.update_layout(

        paper_bgcolor="#F8F4EA",

        plot_bgcolor="#F8F4EA",

        height=450,

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

with right:

    st.subheader(
        "Feature Importance"
    )

    feature_df = pd.DataFrame({

        "Feature": [

            "Credit Score",

            "Loan Amount",

            "Income",

            "Age",

            "Loan Term",

            "Employment Status",

        ],

        "Importance": [

            0.515,

            0.278,

            0.179,

            0.018,

            0.006,

            0.004,

        ],

    })

    fig = px.bar(

        feature_df,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        color_continuous_scale=[
            "#D8C3A5",
            "#C8A97E",
            "#A67C52",
            "#8D6E63",
        ],

    )

    fig.update_layout(

        paper_bgcolor="#F8F4EA",

        plot_bgcolor="#F8F4EA",

        height=450,

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# ============================================================
# CORE CAPABILITIES
# ============================================================

st.subheader(
    "Core Capabilities"
)

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class='fin-card'>
        <h4>Loan Assessment</h4>
        Evaluate applicant eligibility
        and approval probability.
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        """
        <div class='fin-card'>
        <h4>Risk Analytics</h4>
        Analyze applicant risk levels
        and lending recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

c3, c4 = st.columns(2)

with c3:

    st.markdown(
        """
        <div class='fin-card'>
        <h4>Portfolio Insights</h4>
        Explore lending distributions,
        trends and portfolio patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:

    st.markdown(
        """
        <div class='fin-card'>
        <h4>Model Performance</h4>
        Review evaluation metrics,
        reports and feature importance.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
    color:#6D4C41;
    font-size:15px;
    ">
    Developed by Muhammad Farhan
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
    color:#8D6E63;
    font-size:14px;
    ">
    FinSure Analytics • Credit Risk Assessment Platform
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# END OF FILE
# ============================================================