# ============================================================
# FINSURE ANALYTICS
# PORTFOLIO ANALYTICS
# PHASE 12
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
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Portfolio Analytics",

    page_icon=None,

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
# LOAD DATA
# ============================================================

dataset_path = (
    PROJECT_ROOT
    / "data"
    / "loan_data.csv"
)

df = pd.read_csv(
    dataset_path
)

# ============================================================
# HEADER
# ============================================================

st.title(
    "Portfolio Analytics"
)

st.markdown(
    """
    Executive portfolio monitoring and
    lending performance analytics.
    """
)

st.divider()

# ============================================================
# KPI SECTION
# ============================================================

total_applications = len(df)

approved = int(
    df["Loan_Approved"].sum()
)

rejected = (
    total_applications
    - approved
)

approval_rate = (
    approved
    / total_applications
    * 100
)

rejection_rate = (
    rejected
    / total_applications
    * 100
)

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.metric(
        "Applications",
        f"{total_applications:,}"
    )

with k2:

    st.metric(
        "Approved",
        f"{approved:,}"
    )

with k3:

    st.metric(
        "Rejected",
        f"{rejected:,}"
    )

with k4:

    st.metric(
        "Approval Rate",
        f"{approval_rate:.1f}%"
    )

# ============================================================
# APPROVAL ANALYSIS
# ============================================================

st.subheader(
    "Approval Analysis"
)

left, right = st.columns(2)

with left:

    approval_df = pd.DataFrame({

        "Status": [

            "Approved",
            "Rejected",

        ],

        "Count": [

            approved,
            rejected,

        ],

    })

    fig = px.pie(

        approval_df,

        names="Status",

        values="Count",

        color="Status",

        color_discrete_map={

            "Approved":
            "#6E9C6B",

            "Rejected":
            "#C78B4A",

        },

    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:

    approval_bar = px.bar(

        approval_df,

        x="Status",

        y="Count",

        color="Status",

        color_discrete_map={

            "Approved":
            "#6E9C6B",

            "Rejected":
            "#C78B4A",

        },

    )

    approval_bar.update_layout(
        height=450,
        showlegend=False,
    )

    st.plotly_chart(
        approval_bar,
        use_container_width=True,
    )

# ============================================================
# CREDIT SCORE ANALYSIS
# ============================================================

st.subheader(
    "Credit Score Distribution"
)

credit_bins = pd.cut(

    df["Credit_Score"],

    bins=[
        300,
        500,
        650,
        750,
        850,
    ],

    labels=[
        "Poor",
        "Fair",
        "Good",
        "Excellent",
    ],

)

credit_df = (

    credit_bins
    .value_counts()
    .reset_index()

)

credit_df.columns = [

    "Credit Category",

    "Applications",

]

fig = px.bar(

    credit_df,

    x="Credit Category",

    y="Applications",

    color="Credit Category",

)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True,
)
# ============================================================
# INCOME ANALYSIS
# ============================================================

st.subheader(
    "Income Distribution"
)

income_bins = pd.cut(

    df["Income"],

    bins=[
        20000,
        50000,
        80000,
        120000,
        150000,
    ],

    labels=[
        "Low",
        "Medium",
        "High",
        "Very High",
    ],

)

income_df = (

    income_bins
    .value_counts()
    .reset_index()

)

income_df.columns = [

    "Income Group",

    "Applications",

]

fig = px.bar(

    income_df,

    x="Income Group",

    y="Applications",

    color="Income Group",

)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True,
)
# ============================================================
# LOAN AMOUNT ANALYSIS
# ============================================================

st.subheader(
    "Loan Amount Analytics"
)

loan_stats = pd.DataFrame({

    "Metric": [

        "Average",
        "Maximum",
        "Minimum",

    ],

    "Amount": [

        df["Loan_Amount"].mean(),

        df["Loan_Amount"].max(),

        df["Loan_Amount"].min(),

    ],

})

fig = px.bar(

    loan_stats,

    x="Metric",

    y="Amount",

    color="Metric",

)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

# ============================================================
# EMPLOYMENT ANALYSIS
# ============================================================

st.subheader(
    "Employment Profile"
)

employment_df = (

    df["Employment_Status"]
    .value_counts()
    .reset_index()

)

employment_df.columns = [

    "Employment",

    "Count",

]

fig = px.pie(

    employment_df,

    names="Employment",

    values="Count",

    hole=0.55,

)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True,
)
# ============================================================
# APPROVAL RATE BY EMPLOYMENT
# ============================================================

st.subheader(
    "Approval Rate by Employment Status"
)

employment_approval = (

    df.groupby(
        "Employment_Status"
    )["Loan_Approved"]

    .mean()

    .reset_index()

)

employment_approval[
    "Loan_Approved"
] *= 100

fig = px.funnel(

    employment_approval,

    x="Loan_Approved",

    y="Employment_Status",

)

fig.update_layout(

    height=450,

    xaxis_title="Approval Rate (%)",

)

st.plotly_chart(

    fig,

    use_container_width=True,

)

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.subheader(
    "Executive Summary"
)

st.markdown(
    f"""
    <div class='fin-card'>

    <h4>Portfolio Overview</h4>

    <p>
    Total Applications:
    <strong>{total_applications:,}</strong>
    </p>

    <p>
    Approval Rate:
    <strong>{approval_rate:.1f}%</strong>
    </p>

    <p>
    Rejection Rate:
    <strong>{rejection_rate:.1f}%</strong>
    </p>

    <p>
    Average Income:
    <strong>
    ${df['Income'].mean():,.0f}
    </strong>
    </p>

    <p>
    Average Credit Score:
    <strong>
    {df['Credit_Score'].mean():.0f}
    </strong>
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Developed by Muhammad Farhan"
)

st.caption(
    "FinSure Analytics • Credit Risk Assessment Platform"
)

# ============================================================
# END OF FILE
# ============================================================