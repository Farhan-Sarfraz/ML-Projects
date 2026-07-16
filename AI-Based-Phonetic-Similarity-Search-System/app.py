import streamlit as st
import urllib.request
import faiss
import numpy as np
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="PhoneticAI",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #F0F0F0;
    font-family: 'Syne', sans-serif;
    color: #111111;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(99,102,241,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.06) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
    position: relative;
    z-index: 1;
}

.header-wrap {
    text-align: center;
    padding: 50px 20px 30px;
}

.badge {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.4);
    color: #4338CA;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    padding: 6px 18px;
    border-radius: 50px;
    margin-bottom: 20px;
    text-transform: uppercase;
}

.main-title {
    font-size: 64px;
    font-weight: 800;
    line-height: 1.1;
    color: #111111;
    letter-spacing: -2px;
    margin-bottom: 10px;
}

.main-title span {
    background: linear-gradient(90deg, #4F46E5, #0EA5E9, #10B981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sub-title {
    font-size: 18px;
    color: #555555;
    font-weight: 400;
    margin-bottom: 40px;
}

.stats-row {
    display: flex;
    gap: 16px;
    margin-bottom: 30px;
    flex-wrap: wrap;
}

.stat-card {
    flex: 1;
    min-width: 130px;
    background: #FFFFFF;
    border: 1px solid #DDDDDD;
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.3s, box-shadow 0.3s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.stat-card:hover {
    border-color: #4F46E5;
    box-shadow: 0 4px 16px rgba(99,102,241,0.15);
}

.stat-number {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #4F46E5;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 12px;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.how-section {
    display: flex;
    gap: 12px;
    margin: 20px 0 30px;
    flex-wrap: wrap;
}

.how-card {
    flex: 1;
    min-width: 130px;
    background: #FFFFFF;
    border: 1px solid #DDDDDD;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.how-icon { font-size: 28px; margin-bottom: 10px; }

.how-title {
    font-size: 13px;
    font-weight: 700;
    color: #222222;
    margin-bottom: 6px;
}

.how-desc {
    font-size: 12px;
    color: #777777;
    line-height: 1.6;
}

div[data-testid="stForm"] {
    background: #FFFFFF !important;
    border: 1px solid #DDDDDD !important;
    border-radius: 24px !important;
    padding: 30px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;
}

div[data-testid="stTextInput"] input {
    background: #F8F8F8 !important;
    border: 1.5px solid #CCCCCC !important;
    border-radius: 14px !important;
    color: #111111 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 20px !important;
    padding: 16px 20px !important;
    transition: border-color 0.3s !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
    background: #FFFFFF !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #AAAAAA !important;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #4F46E5, #4338CA) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 17px !important;
    padding: 14px !important;
    width: 100% !important;
    margin-top: 10px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.3) !important;
    cursor: pointer !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(79,70,229,0.4) !important;
}

.search-hint {
    font-size: 13px;
    color: #AAAAAA;
    font-family: 'Space Mono', monospace;
    margin-top: 10px;
    text-align: center;
}

.query-info {
    display: flex;
    gap: 14px;
    margin: 20px 0;
    flex-wrap: wrap;
}

.info-pill {
    flex: 1;
    min-width: 180px;
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.info-pill-label {
    font-size: 11px;
    color: #999999;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}

.info-pill-value {
    font-size: 17px;
    font-weight: 700;
    color: #4F46E5;
    font-family: 'Space Mono', monospace;
    word-break: break-all;
}

.results-title {
    font-size: 13px;
    font-weight: 700;
    color: #888888;
    margin: 24px 0 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-row {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 14px;
    padding: 16px 22px;
    margin-bottom: 8px;
    transition: all 0.2s;
    flex-wrap: wrap;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

.result-row:hover {
    background: #F5F3FF;
    border-color: #4F46E5;
    transform: translateX(5px);
    box-shadow: 0 4px 16px rgba(79,70,229,0.1);
}

.best-match {
    background: #EEF2FF !important;
    border-color: #4F46E5 !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.15) !important;
}

.result-rank {
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    color: #AAAAAA;
    min-width: 30px;
}

.result-word {
    font-size: 20px;
    font-weight: 800;
    color: #111111;
    min-width: 160px;
}

.result-phoneme {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #888888;
    flex: 1;
    min-width: 120px;
}

.result-bar-wrap {
    width: 100px;
    height: 6px;
    background: #EEEEEE;
    border-radius: 10px;
    overflow: hidden;
}

.result-bar {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #4F46E5, #0EA5E9);
}

.result-score {
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    color: #4F46E5;
    min-width: 55px;
    text-align: right;
    font-weight: 700;
}

.best-banner {
    background: linear-gradient(135deg, #EEF2FF, #E0F2FE);
    border: 1px solid #4F46E5;
    border-radius: 16px;
    padding: 22px 28px;
    margin-top: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}

.best-banner-label {
    font-size: 12px;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 4px;
}

.best-banner-word {
    font-size: 32px;
    font-weight: 800;
    color: #111111;
}

.best-banner-score {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    color: #4F46E5;
    margin-left: auto;
    font-weight: 700;
}

hr {
    border-color: #DDDDDD !important;
    margin: 30px 0 !important;
}

.footer {
    text-align: center;
    padding: 30px 0 10px;
    color: #BBBBBB;
    font-size: 13px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
}

.footer span { color: #4F46E5; }

div[data-testid="stAlert"] { border-radius: 12px !important; }

h1,h2,h3,h4 { color: #111111 !important; }
label { color: #555555 !important; font-size: 15px !important; }
p { color: #333333; font-size: 15px; }

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD SYSTEM
# =====================================================

@st.cache_resource
def load_system():
    url = "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
    words = []
    phonemes = []
    word_to_phoneme = {}
    response = urllib.request.urlopen(url)
    for line in response:
        line = line.decode("latin-1").strip()
        if line.startswith(";;;"): continue
        parts = line.split()
        if len(parts) < 2: continue
        word = parts[0].lower()
        if "(" in word: continue
        phoneme_sequence = " ".join(parts[1:])
        words.append(word)
        phonemes.append(phoneme_sequence)
        word_to_phoneme[word] = phoneme_sequence
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    vectors = vectorizer.fit_transform(phonemes).toarray().astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return words, phonemes, word_to_phoneme, vectorizer, index


with st.spinner("Loading system — please wait..."):
    words, phonemes, word_to_phoneme, vectorizer, index = load_system()


# =====================================================
# HEADER
# =====================================================

st.markdown(f"""
<div class="header-wrap">
    <div class="main-title">AI Based <span>Phonetic System</span></div>
    <div style="margin-top: 10px; font-size: 18px; font-weight: 700; color: #333333; font-family: 'Syne', sans-serif; letter-spacing: 0.5px;">Muhammad Farhan &nbsp;|&nbsp; 23-CP-74</div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# STATS
# =====================================================

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">{len(words):,}</div>
        <div class="stat-label">Dictionary Words</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">FAISS</div>
        <div class="stat-label">Vector Engine</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">TF-IDF</div>
        <div class="stat-label">Vectorizer</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">Top 10</div>
        <div class="stat-label">Results</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">&lt; 1s</div>
        <div class="stat-label">Search Speed</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# HOW IT WORKS
# =====================================================

st.markdown("""
<div class="how-section">
    <div class="how-card">
        <div class="how-icon">⌨️</div>
        <div class="how-title">Type a Word</div>
        <div class="how-desc">Enter any English word or name</div>
    </div>
    <div class="how-card">
        <div class="how-icon">🔤</div>
        <div class="how-title">Get Phonemes</div>
        <div class="how-desc">CMU Dictionary finds pronunciation</div>
    </div>
    <div class="how-card">
        <div class="how-icon">📐</div>
        <div class="how-title">Vectorize</div>
        <div class="how-desc">TF-IDF converts phonemes to vector</div>
    </div>
    <div class="how-card">
        <div class="how-icon">⚡</div>
        <div class="how-title">FAISS Search</div>
        <div class="how-desc">Searches 125K words instantly</div>
    </div>
    <div class="how-card">
        <div class="how-icon">✅</div>
        <div class="how-title">Results</div>
        <div class="how-desc">Top 10 similar words shown</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# =====================================================
# SEARCH FORM
# =====================================================

with st.form(key="search_form", clear_on_submit=False):
    query = st.text_input(
        label="word",
        label_visibility="collapsed",
        placeholder="Type a word here — Example:  night   |   katherine   |   smith"
    )
    submitted = st.form_submit_button(
        "Find Similar Words",
        use_container_width=True
    )

st.markdown(
    '<div class="search-hint">Press Enter or click the button to search</div>',
    unsafe_allow_html=True
)


# =====================================================
# SEARCH LOGIC
# =====================================================

if submitted:

    if not query or query.strip() == "":
        st.error("Please type a word first.")

    else:
        q = query.lower().strip()

        if q in word_to_phoneme:
            q_phoneme = word_to_phoneme[q]
            source = "CMU Dictionary"
            st.success(f"✅  '{query}' found in CMU Dictionary")
        else:
            primary, secondary = doublemetaphone(q)
            q_phoneme = primary if primary else q
            source = "Double Metaphone"
            st.warning(f"⚠️  '{query}' not found — using Double Metaphone as backup")

        st.markdown(f"""
        <div class="query-info">
            <div class="info-pill">
                <div class="info-pill-label">Word Entered</div>
                <div class="info-pill-value">{query}</div>
            </div>
            <div class="info-pill">
                <div class="info-pill-label">Phoneme</div>
                <div class="info-pill-value">{q_phoneme}</div>
            </div>
            <div class="info-pill">
                <div class="info-pill-label">Source</div>
                <div class="info-pill-value">{source}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        q_vec = vectorizer.transform([q_phoneme]).toarray().astype("float32")
        faiss.normalize_L2(q_vec)
        scores, idxs = index.search(q_vec, 10)

        st.markdown(
            '<div class="results-title">Top 10 Phonetically Similar Words</div>',
            unsafe_allow_html=True
        )

        for i, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            sim = round(score * 100, 2)
            bar_w = max(sim, 3)
            is_best = "best-match" if i == 0 else ""
            rank_label = "🥇" if i == 0 else f"#{i+1}"

            st.markdown(f"""
            <div class="result-row {is_best}">
                <div class="result-rank">{rank_label}</div>
                <div class="result-word">{words[idx]}</div>
                <div class="result-phoneme">{phonemes[idx]}</div>
                <div class="result-bar-wrap">
                    <div class="result-bar" style="width:{bar_w}%"></div>
                </div>
                <div class="result-score">{sim}%</div>
            </div>
            """, unsafe_allow_html=True)

        best_word  = words[idxs[0][0]]
        best_score = round(scores[0][0] * 100, 2)

        st.markdown(f"""
        <div class="best-banner">
            <div>
                <div class="best-banner-label">Best Match Found</div>
                <div class="best-banner-word">{best_word}</div>
            </div>
            <div class="best-banner-score">{best_score}%</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
<div class="footer">

<hr style="border:1px solid rgba(255,255,255,0.15);">

<b>🔍 AI-Based Phonetic Similarity Search System</b><br>

CP-310 • Machine Learning & Artificial Intelligence<br>

Complex Engineering Problem (CEP) • Spring 2026<br>

Developed using CMU Dictionary • Double Metaphone • TF-IDF • FAISS

</div>
""", unsafe_allow_html=True)