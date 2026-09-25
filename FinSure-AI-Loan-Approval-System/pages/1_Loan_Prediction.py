# ============================================================
# FINSURE ANALYTICS
# LOAN PREDICTION INTERFACE
# PHASE 11.2
# ============================================================

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.append(
    str(PROJECT_ROOT)
)

# ============================================================
# BACKEND IMPORTS
# ============================================================

from backend.prediction_engine import (
    PredictionEngine
)

from backend.analytics_engine import (
    AnalyticsEngine
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Loan Prediction",

    page_icon="🏦",

    layout="wide",

)

# ============================================================
# LOAD CSS
# ============================================================

css_file = (
    PROJECT_ROOT
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
# ENGINES
# ============================================================

prediction_engine = (
    PredictionEngine(
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "artifacts",
    )
)

analytics_engine = (
    AnalyticsEngine()
)

# ============================================================
# HEADER
# ============================================================

st.title(
    "Loan Application Assessment"
)

st.markdown(
    """
    Evaluate applicant eligibility using
    predictive risk assessment models and
    lending analytics.
    """
)

st.divider()

# ============================================================
# INPUT FORM
# ============================================================

st.subheader(
    "Applicant Information"
)

left_col, right_col = st.columns(2)

with left_col:

    age = st.number_input(

        "Age",

        min_value=18,

        max_value=100,

        value=35,

    )

    income = st.number_input(

        "Annual Income",

        min_value=0,

        value=80000,

        step=1000,

    )

    employment_status = st.selectbox(

        "Employment Status",

        [

            "Employed",

            "Self-Employed",

            "Unemployed",

        ],

    )

with right_col:

    credit_score = st.number_input(

        "Credit Score",

        min_value=300,

        max_value=850,

        value=700,

    )

    loan_amount = st.number_input(

        "Loan Amount",

        min_value=1000,

        value=15000,

        step=1000,

    )

    loan_term = st.selectbox(

        "Loan Term (Months)",

        [

            12,
            24,
            36,
            48,
            60,

        ],

    )

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "Analyze Application",
    use_container_width=True,
)

# ============================================================
# PROCESS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction_result = (

        prediction_engine.predict(

            age=age,

            income=income,

            credit_score=credit_score,

            loan_amount=loan_amount,

            loan_term=loan_term,

            employment_status=employment_status,

        )

    )

    # --------------------------------------------------------
    # ANALYTICS REPORT
    # --------------------------------------------------------

    report = (

        analytics_engine.build_report(

            prediction_result,

            age,

            income,

            credit_score,

            loan_amount,

        )

    )

    # ========================================================
    # RESULT CARDS
    # ========================================================

    st.divider()

    st.subheader(
        "Decision Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(

            "Application Status",

            report[
                "decision"
            ],

        )

    with col2:

        st.metric(

            "Approval Probability",

            f"{report['approval_probability']}%",

        )

    with col3:

        st.metric(

            "Risk Level",

            report[
                "risk_level"
            ],

        )

    with col4:

        st.metric(

            "Recommendation",

            report[
                "recommendation"
            ],

        )

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    st.subheader(
        "Executive Assessment"
    )

    st.markdown(
        f"""
        <div class='fin-card'>

        <h4>Explanation</h4>

        <p>
        {report['explanation']}
        </p>

        <h4>Recommendation</h4>

        <p>
        {report['recommendation']}
        </p>

        <h4>Key Factors</h4>

        <ul>
            <li>{report['key_factors'][0]}</li>
            <li>{report['key_factors'][1]}</li>
            <li>{report['key_factors'][2]}</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # CHARTS
    # ========================================================

    st.subheader(
        "Analytics Dashboard"
    )

    chart_left, chart_right = st.columns(2)

    # --------------------------------------------------------
    # PIE CHART
    # --------------------------------------------------------

    with chart_left:

        approved_probability = (
            report[
                "approval_probability"
            ]
        )

        rejected_probability = (
            100
            - approved_probability
        )

        pie_df = pd.DataFrame({

            "Status": [

                "Approved",

                "Rejected",

            ],

            "Value": [

                approved_probability,

                rejected_probability,

            ],

        })

        fig = px.pie(

            pie_df,

            names="Status",

            values="Value",

            color="Status",

            color_discrete_map={

                "Approved":
                "#5E9C76",

                "Rejected":
                "#D79A4A",

            },

        )

        fig.update_layout(
            title="Approval Probability Distribution",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # RISK FACTOR CHART
    # --------------------------------------------------------

    with chart_right:

        risk_df = pd.DataFrame({

            "Factor": [

                "Credit Score",

                "Income",

                "Loan Amount",

                "Loan Term",

                "Age",

            ],

            "Score": [

                min(
                    credit_score / 850 * 100,
                    100,
                ),

                min(
                    income / 150000 * 100,
                    100,
                ),

                min(
                    loan_amount / 50000 * 100,
                    100,
                ),

                min(
                    loan_term / 60 * 100,
                    100,
                ),

                min(
                    age / 70 * 100,
                    100,
                ),

            ],

        })

        fig = px.bar(

            risk_df,

            x="Factor",

            y="Score",

            color="Factor",

        )

        fig.update_layout(

            title="Applicant Risk Profile",

            height=450,

            showlegend=False,

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    

# ========================================================
# APPLICANT PROFILE BENCHMARK
# ========================================================

st.subheader(
    "Applicant Profile Benchmark"
)

benchmark_df = pd.DataFrame({

    "Metric": [

        "Credit Score",
        "Income",
        "Loan Burden",

    ],

    "Applicant": [

        credit_score,
        income / 1000,
        loan_amount / 1000,

    ],

    "Benchmark": [

        700,
        80,
        20,

    ],

})

benchmark_long = benchmark_df.melt(

    id_vars="Metric",

    var_name="Profile",

    value_name="Value",

)

fig = px.line(

    benchmark_long,

    x="Metric",

    y="Value",

    color="Profile",

    markers=True,

)

fig.update_layout(

    title="Applicant vs Benchmark Analysis",

    height=500,

    xaxis_title="",

    yaxis_title="Score",

)

st.plotly_chart(

    fig,

    use_container_width=True,

)

# ========================================================
# FOOTER
# ========================================================

st.divider()

st.markdown(
    """
    <div style='text-align:center;'>

    <strong>
    Developed by Muhammad Farhan
    </strong>

    <br>

    FinSure Analytics
    •
    Credit Risk Assessment Platform

    </div>
    """,
    unsafe_allow_html=True,
)