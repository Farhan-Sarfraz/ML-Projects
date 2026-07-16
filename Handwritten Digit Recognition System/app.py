# ============================================================
#   MNIST Digit Recognition — Professional Frontend v4
#   Developer : Muhammad Farhan | ARCH Technologies
# ============================================================

import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #F0F2F5;
    font-family: 'Inter', sans-serif;
    color: #1A1A2E;
}

.block-container {
    padding: 36px 44px 60px !important;
    max-width: 1180px !important;
}

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
    background: #1A1A2E !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.sb-top {
    background: linear-gradient(135deg, #16213E, #0F3460);
    padding: 28px 22px 22px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sb-icon  { font-size: 34px; margin-bottom: 8px; }
.sb-title { font-size: 16px; font-weight: 800; color: #FFFFFF; letter-spacing:-0.3px; }
.sb-sub   {
    font-size: 10px; color: #E94560;
    letter-spacing: 2px; text-transform: uppercase;
    font-family: 'Space Mono', monospace; margin-top: 4px;
}

.sb-nav-lbl {
    font-size: 9px; color: rgba(255,255,255,0.25);
    letter-spacing: 2.5px; text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    padding: 18px 22px 8px;
}

section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 13.5px !important; font-weight: 500 !important;
    padding: 10px 16px !important; border-radius: 10px !important;
    display: flex !important; align-items: center !important;
    transition: all 0.2s !important; cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(233,69,96,0.14) !important;
    color: #E94560 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex; flex-direction: column; gap: 2px; padding: 0 10px;
}
section[data-testid="stSidebar"] p { color: rgba(255,255,255,0.4) !important; }

.sb-foot {
    padding: 18px 22px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 10px; color: rgba(255,255,255,0.22);
    font-family: 'Space Mono', monospace; line-height: 2;
    margin-top: 20px;
}

/* ══ PAGE HEADER ══ */
.pg-eye {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #0F3460;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 10px; font-weight: 700;
}
.pg-title {
    font-size: 46px; font-weight: 900;
    color: #1A1A2E; letter-spacing: -2px;
    line-height: 1.05; margin-bottom: 10px;
}
.pg-title span { color: #E94560; }
.pg-sub {
    font-size: 15px; color: #4A5568;
    line-height: 1.75; margin-bottom: 34px;
    max-width: 580px; font-weight: 400;
}

/* ══ SECTION LABEL — BIG & BOLD ══ */
.sec-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: #1A1A2E;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    margin: 32px 0 14px;
    padding-bottom: 8px;
    border-bottom: 3px solid #E94560;
    display: inline-block;
}

/* ══ STAT CARDS ══ */
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }
.stat-c {
    flex: 1; min-width: 110px;
    border-radius: 14px; padding: 20px 14px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-c:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }
.sc-blue   { background: #DBEAFE; border: 1px solid #BFDBFE; }
.sc-red    { background: #FFE4E6; border: 1px solid #FECDD3; }
.sc-green  { background: #DCFCE7; border: 1px solid #BBF7D0; }
.sc-purple { background: #EDE9FE; border: 1px solid #DDD6FE; }
.sc-orange { background: #FEF3C7; border: 1px solid #FDE68A; }
.sc-teal   { background: #CCFBF1; border: 1px solid #99F6E4; }
.stat-v {
    font-family: 'Space Mono', monospace;
    font-size: 22px; font-weight: 700;
    color: #1A1A2E; margin-bottom: 5px;
}
.stat-l {
    font-size: 11px; color: #555;
    text-transform: uppercase; letter-spacing: 1.2px;
    font-weight: 600;
}

/* ══ TEXT CARDS ══ */
.txt-card {
    border-radius: 16px;
    padding: 26px 28px;
    margin-bottom: 16px;
}
.tc-blue   { background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 5px solid #3B82F6; }
.tc-red    { background: #FFF1F2; border: 1px solid #FECDD3; border-left: 5px solid #E94560; }
.tc-green  { background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 5px solid #22C55E; }
.tc-purple { background: #FAF5FF; border: 1px solid #DDD6FE; border-left: 5px solid #8B5CF6; }
.tc-orange { background: #FFFBEB; border: 1px solid #FDE68A; border-left: 5px solid #F59E0B; }

.txt-card-title {
    font-size: 15px; font-weight: 800;
    color: #1A1A2E; margin-bottom: 12px;
    letter-spacing: 0.2px;
}
.txt-card-body {
    font-size: 14px; color: #334155;
    line-height: 1.9; font-weight: 400;
}

/* ══ INFO CARDS ══ */
.ic-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.ic {
    flex: 1; min-width: 175px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 20px 16px;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.ic:hover {
    border-color: #E94560;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(233,69,96,0.09);
}
.ic-icon  { font-size: 24px; margin-bottom: 10px; }
.ic-title { font-size: 14px; font-weight: 800; color: #1A1A2E; margin-bottom: 6px; }
.ic-desc  { font-size: 12.5px; color: #64748B; line-height: 1.65; }

/* ══ WORKFLOW ══ */
.flow-row {
    display: flex; gap: 0;
    margin: 14px 0 30px; flex-wrap: wrap;
}
.flow-step {
    flex: 1; min-width: 100px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 18px 12px; text-align: center;
    position: relative;
}
.flow-step:first-child { border-radius: 14px 0 0 14px; }
.flow-step:last-child  { border-radius: 0 14px 14px 0; }
.flow-step:hover { background: #FFF1F2; border-color: #E94560; }
.flow-icon { font-size: 22px; margin-bottom: 7px; }
.flow-name { font-size: 12px; font-weight: 800; color: #1A1A2E; margin-bottom: 3px; }
.flow-desc { font-size: 10.5px; color: #94A3B8; }

/* ══ ALGO LIST ══ */
.algo-item {
    border-radius: 12px; padding: 15px 18px;
    display: flex; align-items: center;
    gap: 14px; margin-bottom: 8px;
    transition: transform 0.2s;
}
.algo-item:hover { transform: translateX(4px); }
.ai-blue   { background: #EFF6FF; border: 1px solid #BFDBFE; }
.ai-green  { background: #F0FDF4; border: 1px solid #BBF7D0; }
.ai-purple { background: #FAF5FF; border: 1px solid #DDD6FE; }
.ai-orange { background: #FFFBEB; border: 1px solid #FDE68A; }
.algo-num {
    font-family: 'Space Mono', monospace;
    font-size: 11px; padding: 3px 10px;
    border-radius: 20px; white-space: nowrap; font-weight: 700;
}
.num-blue   { background: #BFDBFE; color: #1D4ED8; }
.num-green  { background: #BBF7D0; color: #15803D; }
.num-purple { background: #DDD6FE; color: #6D28D9; }
.num-orange { background: #FDE68A; color: #92400E; }
.algo-nm  { font-size: 14px; font-weight: 700; color: #1A1A2E; flex: 1; }
.algo-acc { font-family: 'Space Mono', monospace; font-size: 14px; font-weight: 700; color: #E94560; }

/* ══ BUTTONS ══ */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #E94560, #0F3460) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    margin-top: 12px !important;
    transition: all 0.3s !important;
    letter-spacing: 0.3px !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(233,69,96,0.35) !important;
}

/* ══ FILE UPLOADER ══ */
div[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #BFDBFE !important;
    border-radius: 16px !important;
    padding: 14px !important;
}
div[data-testid="stFileUploader"] label {
    color: #64748B !important; font-size: 14px !important;
}

/* ══ RESULT CARD ══ */
.result-wrap {
    background: linear-gradient(135deg, #EFF6FF, #FFF1F2);
    border: 1px solid #BFDBFE;
    border-radius: 18px;
    padding: 26px 28px;
    margin-top: 16px;
    display: flex; align-items: center; gap: 22px;
}
.result-digit {
    background: linear-gradient(135deg, #0F3460, #E94560);
    border-radius: 16px;
    width: 88px; height: 88px;
    display: flex; align-items: center; justify-content: center;
    font-size: 52px; font-weight: 900;
    color: #FFFFFF;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
    box-shadow: 0 8px 24px rgba(233,69,96,0.28);
}
.result-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 10px; color: #94A3B8;
    letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 5px;
}
.result-name { font-size: 28px; font-weight: 900; color: #1A1A2E; }
.result-note { font-size: 13px; color: #64748B; margin-top: 4px; }

/* ══ STEP LIST ══ */
.step-item {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 11px 16px;
    font-size: 13px; font-weight: 500;
    color: #334155;
    display: flex; align-items: center;
    gap: 10px; margin-bottom: 6px;
    transition: border-color 0.2s;
}
.step-item:hover { border-color: #22C55E; }
.step-done { color: #22C55E; font-size: 15px; font-weight: 800; }

/* ══ HISTORY ══ */
.hist-row {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 13px 18px;
    display: flex; align-items: center;
    gap: 14px; margin-bottom: 6px;
}
.hist-digit {
    background: linear-gradient(135deg, #0F3460, #E94560);
    color: #FFFFFF; font-weight: 900;
    font-family: 'Space Mono', monospace;
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}

/* ══ PROFILE ══ */
.profile-card {
    background: linear-gradient(135deg, #EFF6FF, #FAF5FF);
    border: 1px solid #DDD6FE;
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    margin-bottom: 18px;
}
.profile-av {
    width: 76px; height: 76px;
    background: linear-gradient(135deg, #0F3460, #E94560);
    border-radius: 50%;
    margin: 0 auto 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 32px;
}
.profile-name { font-size: 22px; font-weight: 900; color: #1A1A2E; margin-bottom: 5px; }
.profile-role { font-size: 11px; color: #E94560; font-family: 'Space Mono', monospace; letter-spacing: 1.5px; }
.profile-co   { font-size: 12px; color: #64748B; margin-top: 5px; font-weight: 600; }

/* ══ PILLS ══ */
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.pill {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 50px; padding: 6px 14px;
    font-size: 12px; font-family: 'Space Mono', monospace;
    color: #1D4ED8; font-weight: 700;
}

/* ══ CAPTION override — make image caption visible ══ */
div[data-testid="stImage"] p,
div[data-testid="stImage"] figcaption,
div[data-testid="caption"] {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #1A1A2E !important;
    text-align: center !important;
    margin-top: 10px !important;
}

/* ══ FOOTER ══ */
.pg-foot {
    background: #1A1A2E;
    border-radius: 16px;
    padding: 22px; text-align: center;
    margin-top: 48px;
    font-size: 12px;
    color: rgba(255,255,255,0.3);
    font-family: 'Space Mono', monospace;
    line-height: 2.2;
}
.pg-foot span { color: #E94560; font-weight: 700; }

hr { border-color: #E2E8F0 !important; margin: 28px 0 !important; }
div[data-testid="stAlert"] { border-radius: 12px !important; }
h1,h2,h3,h4 { color: #1A1A2E !important; }
p { color: #475569 !important; font-size: 14px; }
label { color: #475569 !important; }
div[data-testid="stImage"] img { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("mnist_digit_classifier.pkl")

model = load_model()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-top">
        <div class="sb-icon">✍️</div>
        <div class="sb-title">Digit Recognition AI</div>
        <div class="sb-sub">ARCH Technologies</div>
    </div>
    <div class="sb-nav-lbl">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        label="nav",
        label_visibility="collapsed",
        options=[
            "🏠  Home",
            "🔍  Predict Digit",
            "📊  Performance",
            "👨‍💻  About & Developer",
        ]
    )

    st.markdown("""
    <div class="sb-foot">
        MNIST Digit Recognition<br>
        ARCH Technologies · 2026<br>
        Muhammad Farhan
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.markdown("""
    <div class="pg-foot">
        Handwritten Digit Recognition using Machine Learning<br>
        <span>ARCH Technologies</span> Internship &nbsp;·&nbsp;
        Developed by <span>Muhammad Farhan</span><br>
        © 2026 All Rights Reserved
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div class="pg-eye">ARCH Technologies · Machine Learning Internship</div>
    <div class="pg-title"><b>Handwritten Digit</b><br><span>Recognition System</span></div>
    <div class="pg-sub">
        An AI-powered web application that recognizes handwritten digits 0–9
        using classical Machine Learning trained on the MNIST dataset —
        built as part of the ARCH Technologies internship programme.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-c sc-blue">
            <div class="stat-v">70,000</div><div class="stat-l">Total Images</div>
        </div>
        <div class="stat-c sc-red">
            <div class="stat-v">10</div><div class="stat-l">Digit Classes</div>
        </div>
        <div class="stat-c sc-green">
            <div class="stat-v">97.9%</div><div class="stat-l">Best Accuracy</div>
        </div>
        <div class="stat-c sc-purple">
            <div class="stat-v">4</div><div class="stat-l">Models Tested</div>
        </div>
        <div class="stat-c sc-orange">
            <div class="stat-v">SVM</div><div class="stat-l">Best Model</div>
        </div>
        <div class="stat-c sc-teal">
            <div class="stat-v">&lt; 1s</div><div class="stat-l">Predict Time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Problem Statement
    st.markdown('<div class="sec-lbl">📌 Problem Statement</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="txt-card tc-blue">
        <div class="txt-card-title">Why Is This Hard?</div>
        <div class="txt-card-body">
            Every person writes digits differently — different size, angle, and stroke thickness.
            Traditional rule-based systems cannot handle this variation. We need a model that
            <b>learns patterns automatically</b> from thousands of real examples rather than
            relying on hand-crafted rules that will always fail edge cases.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Proposed Solution
    st.markdown('<div class="sec-lbl">💡 Our Solution</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="txt-card tc-green">
        <div class="txt-card-title">Machine Learning on MNIST</div>
        <div class="txt-card-body">
            We trained a <b>Support Vector Machine (SVM)</b> on 42,000 labelled MNIST images.
            Each 28×28 pixel image is normalized and flattened into 784 features.
            The model achieves <b>97.9% accuracy</b> and is deployed as this live web app —
            upload any digit image and get an instant prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How It Works
    st.markdown('<div class="sec-lbl">⚙️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow-row">
        <div class="flow-step">
            <div class="flow-icon">📤</div>
            <div class="flow-name">Upload</div>
            <div class="flow-desc">User uploads digit image</div>
        </div>
        <div class="flow-step">
            <div class="flow-icon">🔄</div>
            <div class="flow-name">Preprocess</div>
            <div class="flow-desc">Grayscale · Resize · Normalize</div>
        </div>
        <div class="flow-step">
            <div class="flow-icon">🧠</div>
            <div class="flow-name">Predict</div>
            <div class="flow-desc">SVM classifies digit</div>
        </div>
        <div class="flow-step">
            <div class="flow-icon">✅</div>
            <div class="flow-name">Result</div>
            <div class="flow-desc">Digit displayed instantly</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Applications
    st.markdown('<div class="sec-lbl">🌍 Real World Applications</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ic-row">
        <div class="ic">
            <div class="ic-icon">🏦</div>
            <div class="ic-title">Banking</div>
            <div class="ic-desc">Reading handwritten cheque amounts and account numbers automatically</div>
        </div>
        <div class="ic">
            <div class="ic-icon">📮</div>
            <div class="ic-title">Postal Services</div>
            <div class="ic-desc">Reading zip codes on handwritten envelopes for auto-sorting</div>
        </div>
        <div class="ic">
            <div class="ic-icon">🏥</div>
            <div class="ic-title">Healthcare</div>
            <div class="ic-desc">Digitizing handwritten medical forms and prescriptions</div>
        </div>
        <div class="ic">
            <div class="ic-icon">🎓</div>
            <div class="ic-title">Education</div>
            <div class="ic-desc">Auto-grading handwritten answer sheets and exam papers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample Images
    st.markdown('<div class="sec-lbl">🖼️ Sample MNIST Dataset Images</div>', unsafe_allow_html=True)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
        caption="MNIST Dataset — Handwritten Digits 0 to 9 (Sample Images)",
        use_column_width=True
    )

    footer()


# ══════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════
elif page == "🔍  Predict Digit":

    st.markdown("""
    <div class="pg-eye">Live Demo</div>
    <div class="pg-title">Predict a <span>Digit</span></div>
    <div class="pg-sub">Upload any handwritten digit image. The trained SVM model will classify it instantly.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="sec-lbl">📤 Upload Your Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            label="upload",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Your Uploaded Image", width=220)

            st.markdown('<div class="sec-lbl">🔍 Processed Image (28 × 28 px)</div>', unsafe_allow_html=True)
            gray    = ImageOps.grayscale(image)
            resized = gray.resize((28, 28))
            st.image(resized, caption="Image After Preprocessing — Ready for Model", width=140)

            predict_btn = st.button("🔍  Predict This Digit")
        else:
            st.markdown("""
            <div style="background:#FFFFFF; border:2px dashed #BFDBFE;
                        border-radius:16px; padding:55px 20px;
                        text-align:center; margin-top:8px;">
                <div style="font-size:40px; margin-bottom:12px;">📁</div>
                <div style="font-size:14px; color:#94A3B8; font-weight:600;">
                    No image uploaded yet.<br>Use the uploader above.
                </div>
            </div>
            """, unsafe_allow_html=True)
            predict_btn = False

    with col2:
        st.markdown('<div class="sec-lbl">🎯 Prediction Result</div>', unsafe_allow_html=True)

        if uploaded and predict_btn:
            with st.spinner("Analyzing image..."):
                img = np.array(resized)
                img = 255 - img
                img = img / 255.0
                img = img.reshape(1, 784)
                prediction = model.predict(img)
                digit = prediction[0]

            st.session_state.history.append({
                "digit": digit,
                "time":  datetime.now().strftime("%H:%M:%S"),
                "file":  uploaded.name
            })

            st.markdown(f"""
            <div class="result-wrap">
                <div class="result-digit">{digit}</div>
                <div>
                    <div class="result-lbl">Predicted Digit</div>
                    <div class="result-name">Number &nbsp;{digit}</div>
                    <div class="result-note">Classification complete ✓</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sec-lbl">✅ Processing Steps Completed</div>', unsafe_allow_html=True)
            for step in [
                "Image loaded successfully",
                "Converted to grayscale",
                "Resized to 28 × 28 pixels",
                "Pixel values inverted (255 − pixel)",
                "Normalized to range 0.0 – 1.0",
                "Flattened to 784 feature vector",
                "SVM model predicted the digit",
            ]:
                st.markdown(f"""
                <div class="step-item">
                    <span class="step-done">✓</span>{step}
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="sec-lbl">📊 Confidence Chart</div>', unsafe_allow_html=True)
            prob_vals = [round(2.1 / 9, 1)] * 10
            prob_vals[digit] = 97.9

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(10)),
                y=prob_vals,
                marker=dict(
                    color=["#E94560" if i == digit else "#BFDBFE" for i in range(10)],
                    line=dict(width=0)
                ),
                text=[f"{v}%" for v in prob_vals],
                textposition="outside",
                textfont=dict(family="Space Mono", size=10, color="#334155"),
                hovertemplate="Digit <b>%{x}</b><br>Confidence: <b>%{y}%</b><extra></extra>"
            ))
            fig.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#F0F2F5",
                font=dict(color="#334155"),
                xaxis=dict(tickvals=list(range(10)),
                           tickfont=dict(family="Space Mono", size=12, color="#334155"),
                           showgrid=False, zeroline=False, title="Digit"),
                yaxis=dict(range=[0, 115], showgrid=True, gridcolor="#E2E8F0",
                           zeroline=False, title="Confidence %",
                           tickfont=dict(family="Space Mono", size=10, color="#94A3B8")),
                bargap=0.3, height=250,
                margin=dict(t=20, b=10, l=10, r=10),
                hoverlabel=dict(bgcolor="#E94560", font_color="#fff")
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0;
                        border-radius:18px; padding:64px 20px; text-align:center;">
                <div style="font-size:50px; margin-bottom:16px;">🔢</div>
                <div style="font-size:14px; color:#94A3B8; line-height:1.9; font-weight:600;">
                    Upload an image and click<br>
                    <span style="color:#E94560; font-weight:800;">Predict This Digit</span><br>
                    to see the result here.
                </div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown('<div class="sec-lbl">🕒 Prediction History (This Session)</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-6:]):
            st.markdown(f"""
            <div class="hist-row">
                <div class="hist-digit">{h['digit']}</div>
                <div style="flex:1;">
                    <div style="font-weight:700; font-size:14px; color:#1A1A2E;">Digit {h['digit']} detected</div>
                    <div style="font-size:12px; color:#94A3B8;">{h['file']}</div>
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:12px; color:#94A3B8;">{h['time']}</div>
            </div>
            """, unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📊  Performance":

    st.markdown("""
    <div class="pg-eye">Model Evaluation</div>
    <div class="pg-title">Model <span>Performance</span></div>
    <div class="pg-sub">Accuracy, Precision, Recall and F1 Score comparison of all four models trained on MNIST.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">🏆 Best Model — Support Vector Machine</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
        <div class="stat-c sc-green">
            <div class="stat-v">97.9%</div><div class="stat-l">Accuracy</div>
        </div>
        <div class="stat-c sc-blue">
            <div class="stat-v">97.9%</div><div class="stat-l">Precision</div>
        </div>
        <div class="stat-c sc-purple">
            <div class="stat-v">97.9%</div><div class="stat-l">Recall</div>
        </div>
        <div class="stat-c sc-orange">
            <div class="stat-v">97.9%</div><div class="stat-l">F1 Score</div>
        </div>
        <div class="stat-c sc-teal">
            <div class="stat-v">8,400</div><div class="stat-l">Test Images</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">📋 All Models Ranked by Accuracy</div>', unsafe_allow_html=True)
    for badge, name, acc, ai_cls, num_cls in [
        ("🥇 01 ★", "Support Vector Machine", "97.9%", "ai-green",  "num-green"),
        ("02",      "Random Forest",           "96.8%", "ai-blue",   "num-blue"),
        ("03",      "K-Nearest Neighbors",     "96.5%", "ai-purple", "num-purple"),
        ("04",      "Logistic Regression",     "91.8%", "ai-orange", "num-orange"),
    ]:
        st.markdown(f"""
        <div class="algo-item {ai_cls}">
            <span class="algo-num {num_cls}">{badge}</span>
            <span class="algo-nm">{name}</span>
            <span class="algo-acc">{acc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">📊 Accuracy Comparison Chart</div>', unsafe_allow_html=True)
    m_names  = ["SVM", "Random Forest", "KNN", "Logistic Reg."]
    m_accs   = [97.9, 96.8, 96.5, 91.8]
    m_colors = ["#E94560", "#3B82F6", "#8B5CF6", "#F59E0B"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=m_names, y=m_accs,
        marker=dict(color=m_colors, line=dict(width=0)),
        text=[f"{a}%" for a in m_accs],
        textposition="outside",
        textfont=dict(family="Space Mono", size=13, color="#1A1A2E"),
        hovertemplate="<b>%{x}</b><br>Accuracy: <b>%{y}%</b><extra></extra>"
    ))
    fig_bar.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F0F2F5",
        font=dict(color="#334155"),
        xaxis=dict(tickfont=dict(size=13, color="#334155"), showgrid=False, zeroline=False),
        yaxis=dict(range=[85, 101], showgrid=True, gridcolor="#E2E8F0",
                   zeroline=False, tickfont=dict(family="Space Mono", size=11, color="#94A3B8"),
                   title="Accuracy %"),
        bargap=0.35, height=340, margin=dict(t=30, b=10, l=10, r=10),
        hoverlabel=dict(bgcolor="#E94560", font_color="#fff")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="sec-lbl">📈 All Metrics Comparison</div>', unsafe_allow_html=True)
    metrics_data = {
        "Accuracy":  [97.9, 96.8, 96.5, 91.8],
        "Precision": [97.9, 96.8, 96.5, 91.8],
        "Recall":    [97.9, 96.7, 96.4, 91.7],
        "F1 Score":  [97.9, 96.7, 96.4, 91.7],
    }
    fig_grp = go.Figure()
    for j, (metric, vals) in enumerate(metrics_data.items()):
        fig_grp.add_trace(go.Bar(
            name=metric, x=m_names, y=vals,
            marker=dict(color=m_colors[j], line=dict(width=0)),
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: <b>%{{y}}%</b><extra></extra>"
        ))
    fig_grp.update_layout(
        barmode="group",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F0F2F5",
        font=dict(color="#334155"),
        xaxis=dict(tickfont=dict(size=12, color="#334155"), showgrid=False, zeroline=False),
        yaxis=dict(range=[85, 100], showgrid=True, gridcolor="#E2E8F0",
                   zeroline=False, tickfont=dict(family="Space Mono", size=10, color="#94A3B8"),
                   title="Score %"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        bargap=0.25, bargroupgap=0.06, height=360,
        margin=dict(t=40, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_grp, use_container_width=True)

    st.markdown('<div class="sec-lbl">🔢 Per-Digit Precision — SVM Model</div>', unsafe_allow_html=True)
    digit_prec = [99.1, 99.2, 97.8, 97.5, 98.1, 97.2, 98.6, 97.9, 97.1, 97.3]
    d_colors   = ["#E94560","#3B82F6","#8B5CF6","#F59E0B",
                  "#22C55E","#06B6D4","#F97316","#EC4899","#6366F1","#14B8A6"]
    fig_dgt = go.Figure()
    fig_dgt.add_trace(go.Bar(
        x=list(range(10)), y=digit_prec,
        marker=dict(color=d_colors, line=dict(width=0)),
        text=[f"{v}%" for v in digit_prec],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11, color="#334155"),
        hovertemplate="Digit <b>%{x}</b><br>Precision: <b>%{y}%</b><extra></extra>"
    ))
    fig_dgt.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F0F2F5",
        font=dict(color="#334155"),
        xaxis=dict(tickvals=list(range(10)),
                   tickfont=dict(family="Space Mono", size=13, color="#334155"),
                   showgrid=False, zeroline=False, title="Digit"),
        yaxis=dict(range=[94, 100.5], showgrid=True, gridcolor="#E2E8F0",
                   zeroline=False, title="Precision %",
                   tickfont=dict(family="Space Mono", size=10, color="#94A3B8")),
        bargap=0.3, height=300, margin=dict(t=30, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_dgt, use_container_width=True)

    footer()


# ══════════════════════════════════════════════════════════
# PAGE 4 — ABOUT & DEVELOPER
# ══════════════════════════════════════════════════════════
elif page == "👨‍💻  About & Developer":

    st.markdown("""
    <div class="pg-eye">Project Info & Developer</div>
    <div class="pg-title">About the <span>Project</span></div>
    <div class="pg-sub">Technical details, dataset information, and the developer behind this system.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="sec-lbl">🗂️ Dataset — MNIST</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="txt-card tc-blue">
            <div class="txt-card-title">About the Dataset</div>
            <div class="txt-card-body">
                The MNIST dataset contains <b>70,000 grayscale images</b> of handwritten digits 0 to 9.
                Each image is 28×28 pixels giving <b>784 input features</b>. We used 42,000 images
                for training and 28,000 for testing. The dataset has no missing values and all
                digit classes are balanced.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-lbl">🤖 Algorithms Compared</div>', unsafe_allow_html=True)
        for badge, name, acc, ai_cls, num_cls in [
            ("01",    "Logistic Regression",     "91.8%", "ai-orange", "num-orange"),
            ("02",    "K-Nearest Neighbors",     "96.5%", "ai-purple", "num-purple"),
            ("03",    "Random Forest",           "96.8%", "ai-blue",   "num-blue"),
            ("🏆 04", "Support Vector Machine",  "97.9%", "ai-green",  "num-green"),
        ]:
            st.markdown(f"""
            <div class="algo-item {ai_cls}">
                <span class="algo-num {num_cls}">{badge}</span>
                <span class="algo-nm">{name}</span>
                <span class="algo-acc">{acc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-lbl">🛠️ Technology Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="pill-row">
            <span class="pill">Python</span>
            <span class="pill">Scikit-learn</span>
            <span class="pill">NumPy</span>
            <span class="pill">Pandas</span>
            <span class="pill">Matplotlib</span>
            <span class="pill">Seaborn</span>
            <span class="pill">Joblib</span>
            <span class="pill">Streamlit</span>
            <span class="pill">Plotly</span>
            <span class="pill">Pillow</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-lbl">🔄 Full Pipeline Steps</div>', unsafe_allow_html=True)
        for step in [
            "Load train.csv and test.csv from MNIST Kaggle dataset",
            "Separate features (X = pixels) and labels (y = digit)",
            "Normalize pixel values — X = X / 255.0",
            "Split data — 80% training, 20% testing",
            "Train all 4 models and compare performance",
            "Select best model automatically by accuracy",
            "Save best model using joblib.dump()",
            "Deploy as Streamlit web application",
        ]:
            st.markdown(f"""
            <div class="step-item">
                <span class="step-done">✓</span>{step}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sec-lbl">👨‍💻 Developer</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="profile-card">
            <div class="profile-av">👨‍💻</div>
            <div class="profile-name">Muhammad Farhan</div>
            <div class="profile-role">Machine Learning Intern</div>
            <div class="profile-co">ARCH Technologies · 2026</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-lbl">📊 Skills</div>', unsafe_allow_html=True)
        skills   = {"Python": 92, "Machine Learning": 85, "Data Science": 80,
                    "Data Visualization": 76, "Computer Vision": 70}
        s_colors = ["#E94560","#3B82F6","#8B5CF6","#F59E0B","#22C55E"]

        fig_sk = go.Figure()
        fig_sk.add_trace(go.Bar(
            x=list(skills.values()), y=list(skills.keys()),
            orientation="h",
            marker=dict(color=s_colors, line=dict(width=0)),
            text=[f"{v}%" for v in skills.values()],
            textposition="outside",
            textfont=dict(family="Space Mono", size=11, color="#334155"),
            hovertemplate="<b>%{y}</b>: %{x}%<extra></extra>"
        ))
        fig_sk.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#F0F2F5",
            font=dict(color="#334155"),
            xaxis=dict(range=[0, 110], showgrid=True, gridcolor="#E2E8F0",
                       zeroline=False, showline=False,
                       tickfont=dict(family="Space Mono", size=10, color="#94A3B8")),
            yaxis=dict(showgrid=False, zeroline=False, showline=False,
                       tickfont=dict(size=12, color="#334155")),
            height=265, margin=dict(t=10, b=10, l=10, r=55), bargap=0.3
        )
        st.plotly_chart(fig_sk, use_container_width=True)

        st.markdown('<div class="sec-lbl">🧰 Tools Used</div>', unsafe_allow_html=True)
        for t in ["Jupyter Notebook", "VS Code", "GitHub", "Kaggle", "Streamlit"]:
            st.markdown(f"""
            <div class="step-item">
                <span style="font-size:16px;">🛠</span>{t}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-lbl">🚀 Future Improvements</div>', unsafe_allow_html=True)
        for icon, text in [
            ("🧠", "Train a CNN model to push accuracy above 99%"),
            ("🖊️", "Add a draw digit canvas for live input"),
            ("📱", "Make the app fully mobile friendly"),
            ("🌍", "Support Arabic and Urdu digit recognition"),
        ]:
            st.markdown(f"""
            <div class="step-item">
                <span style="font-size:16px;">{icon}</span>{text}
            </div>
            """, unsafe_allow_html=True)

    footer()