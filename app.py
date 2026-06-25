"""
app.py — Streamlit NIDS Web Application
=========================================
Deploys the trained NIDS models as an interactive web dashboard.

Tabs:
  1. 🏠 Home        — Project overview and class distribution
  2. 🔍 Live Predict — Predict on a single traffic record via form
  3. 📁 Batch Predict — Upload a CSV and predict on all rows
  4. 📊 Model Metrics — View evaluation plots and comparison charts
  5. ℹ️ About        — Dataset and methodology summary
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Windows consoles default to a non-UTF-8 codepage, which raises
# UnicodeEncodeError on any emoji printed to stdout/stderr (this process and
# everything it imports, e.g. predict.py's loading messages). Force UTF-8 so
# a console print can never crash model loading.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS — Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Theme / CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
    --accent:#6366f1;
    --accent2:#22d3ee;
    --ink:#0f172a;
    --muted:#64748b;
    --card-radius:14px;
    --shadow-sm:0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.08);
    --shadow-md:0 10px 28px -8px rgba(15,23,42,.18);
    --shadow-lg:0 20px 45px -12px rgba(99,102,241,.35);
}

html, body, [class*="css"]{ font-family:'Inter', sans-serif; }
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3{
    font-family:'Poppins', sans-serif !important;
    font-weight:700 !important;
    letter-spacing:-0.01em;
}

/* ── animated entrance ───────────────────────────────────────────────── */
@keyframes fadeInUp{ from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:translateY(0);} }
@keyframes pulseGlow{ 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.35);} 50%{box-shadow:0 0 0 10px rgba(239,68,68,0);} }
@keyframes shimmer{ 0%{background-position:-400px 0;} 100%{background-position:400px 0;} }
@keyframes floatIcon{ 0%,100%{transform:translateY(0);} 50%{transform:translateY(-4px);} }

.main .block-container{ animation:fadeInUp .45s ease both; }
[data-testid="stMainBlockContainer"]{ padding-top:1.25rem !important; }
[data-testid="stHeader"]{ height:2.75rem !important; min-height:2.75rem !important; }

/* ── app background ──────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main{
    background:radial-gradient(circle at 0% 0%, #eef2ff 0%, #ffffff 28%) ;
}

/* ── sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"]{
    background:linear-gradient(195deg, #0f172a 0%, #1e1b4b 55%, #312e81 100%);
}
[data-testid="stSidebar"] *{ color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div{
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.18);
    border-radius:10px;
    color:#fff !important;
}
[data-testid="stSidebar"] hr{ border-color:rgba(255,255,255,.15); }

/* ── buttons ──────────────────────────────────────────────────────────── */
.stButton button, .stFormSubmitButton button, [data-testid="stDownloadButton"] button{
    border-radius:10px !important;
    font-weight:600 !important;
    border:none !important;
    transition:transform .15s ease, box-shadow .25s ease !important;
    box-shadow:var(--shadow-sm);
}
.stButton button:hover, .stFormSubmitButton button:hover, [data-testid="stDownloadButton"] button:hover{
    transform:translateY(-2px);
    box-shadow:var(--shadow-md);
}
.stFormSubmitButton button[kind="primary"]{
    background:linear-gradient(135deg, var(--accent), #8b5cf6) !important;
}

/* ── tabs ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{
    gap:10px;
    background:#f1f5f9;
    padding:8px;
    border-radius:14px;
    margin-bottom:6px;
}
.stTabs [data-baseweb="tab"]{
    border-radius:10px !important;
    font-weight:600;
    color:var(--muted);
    padding:10px 18px !important;
    height:auto !important;
    transition:all .2s ease;
}
.stTabs [data-baseweb="tab"]:hover{ background:rgba(99,102,241,.08); color:var(--accent); }
.stTabs [aria-selected="true"]{
    background:#fff !important;
    color:var(--accent) !important;
    box-shadow:var(--shadow-sm);
}

/* ── metrics ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"]{
    background:#fff;
    border:1px solid #eef0f4;
    border-radius:var(--card-radius);
    padding:14px 16px;
    box-shadow:var(--shadow-sm);
    transition:transform .2s ease, box-shadow .2s ease;
}
[data-testid="stMetric"]:hover{ transform:translateY(-3px); box-shadow:var(--shadow-md); }
[data-testid="stMetricValue"]{ font-family:'Poppins',sans-serif; font-weight:700; color:var(--ink); }

/* ── generic interactive card (used for our own HTML cards) ─────────────── */
.fine-card{
    border-radius:var(--card-radius);
    transition:transform .22s ease, box-shadow .22s ease;
    animation:fadeInUp .5s ease both;
}
.fine-card:hover{ transform:translateY(-4px) scale(1.012); box-shadow:var(--shadow-md); }

/* ── forms / inputs ───────────────────────────────────────────────────── */
[data-testid="stForm"]{
    background:#fff;
    border:1px solid #eef0f4;
    border-radius:18px;
    padding:1.6rem 1.6rem .6rem;
    box-shadow:var(--shadow-sm);
}
.stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div{
    border-radius:9px !important;
    transition:box-shadow .2s ease, border-color .2s ease !important;
}
.stTextInput input:focus, .stNumberInput input:focus{
    box-shadow:0 0 0 3px rgba(99,102,241,.18) !important;
    border-color:var(--accent) !important;
}
.stSlider [data-baseweb="slider"] > div > div{ background:var(--accent) !important; }

/* ── file uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"]{
    border-radius:14px !important;
    border:2px dashed #c7d2fe !important;
    background:#f8fafc !important;
    transition:border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:var(--accent) !important;
    background:#eef2ff !important;
}

/* ── expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"]{
    border-radius:14px !important;
    border:1px solid #eef0f4 !important;
    box-shadow:var(--shadow-sm);
}

/* ── images (plots) ───────────────────────────────────────────────────── */
[data-testid="stImage"] img{
    border-radius:12px;
    box-shadow:var(--shadow-sm);
    transition:transform .25s ease, box-shadow .25s ease;
}
[data-testid="stImage"] img:hover{ transform:scale(1.012); box-shadow:var(--shadow-md); }

/* ── dataframes ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"]{ border-radius:12px; overflow:hidden; box-shadow:var(--shadow-sm); }

/* ── markdown tables (About tab) ──────────────────────────────────────── */
.stMarkdown table{ border-collapse:separate; border-spacing:0; border-radius:12px; overflow:hidden; box-shadow:var(--shadow-sm); }
.stMarkdown table thead tr{ background:linear-gradient(135deg, var(--accent), #8b5cf6); }
.stMarkdown table thead th{ color:#fff !important; }
.stMarkdown table tbody tr:nth-child(even){ background:#f8fafc; }
.stMarkdown table tbody tr:hover{ background:#eef2ff; }

/* ── alerts ───────────────────────────────────────────────────────────── */
[data-testid="stAlert"]{ border-radius:12px; box-shadow:var(--shadow-sm); }

/* ── scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:#c7d2fe; border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_COLORS  = {
    "Normal": "#2ecc71",
    "DoS":    "#e74c3c",
    "Probe":  "#f39c12",
    "R2L":    "#9b59b6",
    "U2R":    "#1abc9c",
}

def _fade_color(hex_color, alpha=0.35):
    hex_color = (hex_color or "#888888").lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        return f"rgba(136,136,136,{alpha})"
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR     = "plots"

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

PROTOCOL_OPTS = ["tcp", "udp", "icmp"]
SERVICE_OPTS  = [
    "http", "ftp", "smtp", "ssh", "dns", "ftp_data", "telnet",
    "pop_3", "imap4", "https", "other"
]
FLAG_OPTS     = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3", "OTH"]


# ─── Cached Model Loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models… this may take a moment.")
def load_predictor(model_type="both"):
    try:
        from predict import NIDSPredictor
        return NIDSPredictor(model_type=model_type)
    except Exception as e:
        st.error(f"❌ Could not load models: {e}\n\nPlease run `python main.py` first.")
        return None


# ─── Helper: load plot image ──────────────────────────────────────────────────
def load_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:6px 0 14px;'>
        <div style='font-size:46px;animation:floatIcon 3s ease-in-out infinite;'>🛡️</div>
        <div style='font-family:Poppins,sans-serif;font-weight:700;font-size:22px;
                    color:#fff;margin-top:2px;'>NIDS Dashboard</div>
        <div style='font-size:13px;color:#a5b4fc;margin-top:2px;'>Network Intrusion Detection System</div>
        <div style='font-size:12px;color:#818cf8;font-style:italic;'>Random Forest + LSTM · NSL-KDD</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    model_choice = st.selectbox(
        "🧠 Active Model",
        options=["Both (Ensemble)", "Random Forest", "LSTM"],
        index=0
    )
    model_map = {"Both (Ensemble)": "both", "Random Forest": "rf", "LSTM": "lstm"}
    active_model = model_map[model_choice]

    st.divider()
    st.markdown("""
    <div class='fine-card' style='background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.35);
                border-radius:12px;padding:10px 12px;font-size:12.5px;'>
        🟢 <b>Models cached</b> &mdash; inference runs in real time once loaded.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("🎓 Final Year Project")
    st.caption("Design & Implementation of a NIDS")

predictor = load_predictor(active_model)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_home, tab_live, tab_batch, tab_metrics, tab_about = st.tabs([
    "🏠 Home", "🔍 Live Predict", "📁 Batch Predict", "📊 Model Metrics", "ℹ️ About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class='fine-card' style='background:linear-gradient(120deg,#312e81 0%,#4338ca 45%,#6366f1 100%);
                padding:34px 36px;border-radius:20px;color:#fff;margin-bottom:22px;
                box-shadow:0 20px 45px -12px rgba(67,56,202,.45);'>
        <div style='font-size:38px;line-height:1;'>🛡️</div>
        <div style='font-family:Poppins,sans-serif;font-weight:800;font-size:30px;margin-top:6px;'>
            Network Intrusion Detection System
        </div>
        <div style='font-size:15.5px;color:#e0e7ff;margin-top:8px;max-width:760px;line-height:1.55;'>
            An intelligent system that classifies network traffic into <b>Normal</b> or one of four
            attack categories using <b>Random Forest</b> and <b>LSTM</b> deep learning models trained
            on the <b>NSL-KDD</b> benchmark dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Records", "125,973")
    col2.metric("Test Records",     "22,544")
    col3.metric("Feature Classes",  "5")
    col4.metric("Input Features",   "40 (selected)")

    st.divider()

    # Attack class cards
    st.subheader("🏷️ Attack Categories")
    cols = st.columns(5)
    descriptions = {
        "Normal": "Legitimate traffic — no attack detected.",
        "DoS":    "Denial of Service — floods the target to deny service.",
        "Probe":  "Surveillance / scanning to gather information.",
        "R2L":    "Remote to Local — unauthorised access from remote machine.",
        "U2R":    "User to Root — privilege escalation attacks.",
    }
    icons = {"Normal": "✅", "DoS": "🔴", "Probe": "🟡", "R2L": "🟣", "U2R": "🔵"}
    for col, cls in zip(cols, CLASS_NAMES):
        with col:
            st.markdown(
                f"""<div class='fine-card' style='background:{CLASS_COLORS[cls]}1a;border:1px solid {CLASS_COLORS[cls]}40;
                border-left:4px solid {CLASS_COLORS[cls]};padding:14px;border-radius:12px;min-height:118px;'>
                <div style='font-size:20px;'>{icons[cls]}</div>
                <b style='font-size:14.5px;'>{cls}</b>
                <div style='font-size:12px;color:#475569;margin-top:4px;line-height:1.4;'>{descriptions[cls]}</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader("📈 NSL-KDD Class Distribution (Train Set)")

    with st.container(border=True):
        # Approximate distribution from the NSL-KDD dataset
        dist = {
            "Normal": 67343, "DoS": 45927, "Probe": 11656, "R2L": 995, "U2R": 52
        }
        fig = px.bar(
            x=list(dist.keys()), y=list(dist.values()),
            color=list(dist.keys()),
            color_discrete_map=CLASS_COLORS,
            labels={"x": "Class", "y": "Sample Count"},
            title="Training Set Class Distribution",
            text_auto=True,
        )
        fig.update_layout(
            showlegend=False, height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.title("🔍 Live Traffic Prediction")
    st.info("Fill in the network connection features below and click **Predict** to classify the traffic.")

    if predictor is None:
        st.warning("Models not loaded. Please train models first.")
    else:
        with st.form("predict_form"):
            st.subheader("Connection Basics")
            c1, c2, c3, c4 = st.columns(4)
            duration      = c1.number_input("Duration (s)",     min_value=0, value=0)
            protocol_type = c2.selectbox("Protocol Type",       PROTOCOL_OPTS)
            service       = c3.selectbox("Service",             SERVICE_OPTS)
            flag          = c4.selectbox("Flag",                FLAG_OPTS)

            st.subheader("Traffic Volume")
            c1, c2, c3, c4 = st.columns(4)
            src_bytes      = c1.number_input("Src Bytes",       min_value=0, value=215)
            dst_bytes      = c2.number_input("Dst Bytes",       min_value=0, value=45076)
            land           = c3.selectbox("Land",               [0, 1])
            wrong_fragment = c4.number_input("Wrong Fragments", min_value=0, value=0)

            st.subheader("Login & Access Features")
            c1, c2, c3, c4 = st.columns(4)
            logged_in          = c1.selectbox("Logged In",          [0, 1], index=1)
            num_failed_logins  = c2.number_input("Failed Logins",   min_value=0, value=0)
            num_compromised    = c3.number_input("Compromised",     min_value=0, value=0)
            root_shell         = c4.selectbox("Root Shell",         [0, 1])

            st.subheader("Connection Rate Features")
            c1, c2, c3, c4 = st.columns(4)
            count           = c1.number_input("Count",             min_value=0, value=1)
            srv_count       = c2.number_input("Srv Count",         min_value=0, value=1)
            serror_rate     = c3.slider("SError Rate",             0.0, 1.0, 0.0)
            rerror_rate     = c4.slider("RError Rate",             0.0, 1.0, 0.0)

            c1, c2, c3, c4 = st.columns(4)
            same_srv_rate   = c1.slider("Same Srv Rate",           0.0, 1.0, 1.0)
            diff_srv_rate   = c2.slider("Diff Srv Rate",           0.0, 1.0, 0.0)
            dst_host_count  = c3.number_input("Dst Host Count",    min_value=0, value=15)
            dst_host_srv_count = c4.number_input("Dst Host Srv Count", min_value=0, value=15)

            submitted = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)

        if submitted:
            record = {
                "duration": duration, "protocol_type": protocol_type,
                "service": service, "flag": flag,
                "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                "land": land, "wrong_fragment": wrong_fragment,
                "urgent": 0, "hot": 1,
                "num_failed_logins": num_failed_logins, "logged_in": logged_in,
                "num_compromised": num_compromised, "root_shell": root_shell,
                "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
                "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
                "is_host_login": 0, "is_guest_login": 0,
                "count": count, "srv_count": srv_count,
                "serror_rate": serror_rate, "srv_serror_rate": serror_rate,
                "rerror_rate": rerror_rate, "srv_rerror_rate": rerror_rate,
                "same_srv_rate": same_srv_rate, "diff_srv_rate": diff_srv_rate,
                "srv_diff_host_rate": 0.0, "dst_host_count": dst_host_count,
                "dst_host_srv_count": dst_host_srv_count,
                "dst_host_same_srv_rate": same_srv_rate,
                "dst_host_diff_srv_rate": diff_srv_rate,
                "dst_host_same_src_port_rate": 0.07,
                "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": serror_rate,
                "dst_host_srv_serror_rate": serror_rate,
                "dst_host_rerror_rate": rerror_rate,
                "dst_host_srv_rerror_rate": rerror_rate,
            }

            with st.spinner("Running inference..."):
                result = predictor.predict_single(record)

            cls        = result["prediction"]
            confidence = result["confidence"]
            color      = CLASS_COLORS.get(cls, "#888")
            icon       = "✅" if cls == "Normal" else "🚨"
            uncertain  = result.get("is_uncertain", False)
            runner_up  = result.get("runner_up", {})
            rf_vote    = result.get("rf_vote")
            lstm_vote  = result.get("lstm_vote")

            # ── Uncertainty override ──────────────────────────────────────────
            # If models disagree OR confidence < 70%, treat as SUSPICIOUS
            models_disagree = (rf_vote and lstm_vote and rf_vote != lstm_vote)
            show_warning    = uncertain or models_disagree

            if show_warning and cls == "Normal":
                display_cls   = "⚠️ SUSPICIOUS (Low Confidence)"
                display_color = "#f39c12"
                display_icon  = "⚠️"
            else:
                display_cls   = cls
                display_color = color
                display_icon  = icon

            # ── Main verdict card ─────────────────────────────────────────────
            is_alarming = (result["is_attack"] or show_warning)
            pulse_style = "animation:pulseGlow 1.8s infinite;" if is_alarming else ""
            st.markdown(f"""
            <div class='fine-card' style='background:{display_color}1f;border:2px solid {display_color};
            padding:24px;border-radius:14px;text-align:center;margin:15px 0;{pulse_style}'>
            <h2 style='color:{display_color};margin:0'>{display_icon} {display_cls}</h2>
            <p style='margin:10px 0 6px;font-size:18px'>
                Ensemble Confidence: <b>{confidence*100:.1f}%</b>
            </p>
            <div style='background:{display_color}25;border-radius:8px;height:10px;
                        max-width:420px;margin:0 auto;overflow:hidden;'>
                <div style='background:{display_color};height:100%;border-radius:8px;
                            width:{confidence*100:.1f}%;transition:width 1s ease;'></div>
            </div>
            <p style='margin:12px 0 0;font-size:14px;color:#555'>
                {"⚠️ Models disagree — treat as suspicious." if models_disagree
                 else ("⚠️ Low confidence — result uncertain." if uncertain
                 else ("🚨 Intrusion Detected!" if result["is_attack"]
                 else "✅ Traffic appears normal."))}
            </p>
            </div>""", unsafe_allow_html=True)

            # ── Model disagreement alert ──────────────────────────────────────
            if models_disagree:
                ru_class = runner_up.get("class", "")
                ru_prob  = runner_up.get("prob", 0) * 100
                st.warning(
                    f"🔀 Model Disagreement Detected  "
                    f"| RF: {rf_vote}  |  LSTM: {lstm_vote}  "
                    f"| Runner-up: {ru_class} ({ru_prob:.1f}%). "
                    f"Consider RF vote as primary for tabular data."
                )

            # ── Per-model vote badges ─────────────────────────────────────────
            if rf_vote or lstm_vote:
                st.markdown("#### 🗳️ Individual Model Votes")
                vc1, vc2 = st.columns(2)
                if rf_vote:
                    rc = CLASS_COLORS.get(rf_vote, "#888")
                    vc1.markdown(
                        f"<div class='fine-card' style='background:{rc}1f;border:2px solid {rc};"
                        f"padding:14px;border-radius:12px;text-align:center'>"
                        f"<b>🌲 Random Forest</b><br>"
                        f"<span style='font-size:20px;color:{rc}'><b>{rf_vote}</b></span>"
                        f"</div>", unsafe_allow_html=True
                    )
                if lstm_vote:
                    lc = CLASS_COLORS.get(lstm_vote, "#888")
                    vc2.markdown(
                        f"<div class='fine-card' style='background:{lc}1f;border:2px solid {lc};"
                        f"padding:14px;border-radius:12px;text-align:center'>"
                        f"<b>🧠 LSTM</b><br>"
                        f"<span style='font-size:20px;color:{lc}'><b>{lstm_vote}</b></span>"
                        f"</div>", unsafe_allow_html=True
                    )
                st.markdown("")

            # ── Ensemble probability chart ────────────────────────────────────
            st.markdown("#### 📊 Ensemble Class Probabilities")
            probs = result["class_probs"]

            # Always display bars in canonical order: Normal, DoS, Probe, R2L, U2R
            # regardless of how the label encoder sorted them internally.
            display_order = [c for c in CLASS_NAMES if c in probs]
            disp_values   = [probs[c] for c in display_order]

            # The predicted class is the one with the HIGHEST probability in the chart
            chart_winner  = display_order[int(np.argmax(disp_values))]

            # Build simple solid-color bars (no per-bar line-width list —
            # older plotly versions reject list values for marker_line_width)
            fig = go.Figure()
            # Draw non-winner bars first, then winner on top for emphasis
            for i, (c, v) in enumerate(zip(display_order, disp_values)):
                is_winner = (c == chart_winner)

                # Get the base color for this class
                base_color = CLASS_COLORS.get(c, "#888888")

                # Use a faded color for non-winners
                fill_color = base_color if is_winner else _fade_color(base_color, 0.35)

                # Add the bar
                fig.add_trace(
                    go.Bar(
                        x=[c],
                        y=[v],
                        marker_color=fill_color,
                        marker_line_color=base_color,
                        marker_line_width=3 if is_winner else 1,
                        text=[f"<b>{v*100:.1f}%</b>" if is_winner else f"{v*100:.1f}%"],
                        textposition="outside",
                        showlegend=False,
                        name=c,
                    )
                )

            fig.update_layout(
                barmode="group",
                yaxis=dict(
                    range=[0, min(max(disp_values) * 1.35, 1.15)],
                    title="Probability",
                    tickformat=".0%",
                    gridcolor="#eee",
                ),
                xaxis=dict(
                    title="Attack Class",
                    categoryorder="array",
                    categoryarray=display_order,
                ),
                height=400,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=13),
                margin=dict(t=20, b=40),
            )

            # Display the Plotly chart
            st.plotly_chart(fig, width="stretch")

            # Sanity check: ensure the displayed probabilities agree with the predicted class
            if chart_winner != cls:
                st.error(
                    f"⚠️ Chart winner is **{chart_winner}** "
                    f"({probs.get(chart_winner, 0) * 100:.1f}%), "
                    f"but the predicted verdict is **{cls}**.\n\n"
                    "This usually indicates that the class labels and probability "
                    "columns are not aligned. Check your `class_names` ordering and "
                    "the model output before retraining."
                )

            # ── Side-by-side per-model breakdown ─────────────────────────────
            rf_p   = result.get("rf_probs",   {})
            lstm_p = result.get("lstm_probs", {})
            if rf_p and lstm_p:
                with st.expander("🔍 Per-Model Probability Breakdown"):
                    bc1, bc2 = st.columns(2)
                    for col, model_p, mdl_title in [
                        (bc1, rf_p,   "🌲 Random Forest"),
                        (bc2, lstm_p, "🧠 LSTM"),
                    ]:
                        m_order  = [c for c in CLASS_NAMES if c in model_p]
                        m_values = [model_p[c] for c in m_order]
                        m_winner = m_order[int(np.argmax(m_values))]
                        fig2 = go.Figure()
                        for c, v in zip(m_order, m_values):
                            fig2.add_trace(go.Bar(
                                x=[c], y=[v],
                                marker_color=CLASS_COLORS.get(c, "#888") if c == m_winner
                                             else _fade_color(CLASS_COLORS.get(c, "#888"), 0.35),
                                marker_line_color=CLASS_COLORS.get(c, "#888"),
                                marker_line_width=2 if c == m_winner else 0,
                                text=[f"<b>{v*100:.1f}%</b>" if c == m_winner
                                      else f"{v*100:.1f}%"],
                                textposition="outside",
                                showlegend=False,
                                name=c,
                            ))
                        fig2.update_layout(
                            title=dict(text=mdl_title, font=dict(size=13)),
                            barmode="group",
                            yaxis=dict(range=[0, min(max(m_values)*1.35, 1.15)],
                                       tickformat=".0%", gridcolor="#eee"),
                            height=300,
                            showlegend=False,
                            margin=dict(t=45, b=20),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        col.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.title("📁 Batch Prediction")
    st.markdown(
        "Upload a **CSV file** containing NSL-KDD feature columns. "
        "The system will classify every row and return results as a downloadable CSV."
    )

    if predictor is None:
        st.warning("Models not loaded. Please train models first.")
    else:
        uploaded = st.file_uploader(
            "Upload CSV (41 NSL-KDD feature columns, no header label column)",
            type=["csv", "txt"]
        )

        if uploaded is not None:
            try:
                # ── Smart file reading ────────────────────────────────────────
                # Peek at the first line to decide whether there is a header row.
                # NSL-KDD .txt files have NO header; the first cell will be a
                # number (duration). Named CSVs have a text header like "duration".
                import io
                raw_bytes = uploaded.read()
                first_line = raw_bytes.split(b"\n")[0].decode("utf-8", errors="ignore")
                first_cell = first_line.split(",")[0].strip().strip('"')

                if first_cell.lstrip("-").replace(".", "").isdigit():
                    # No header row — read without header
                    df_input = pd.read_csv(io.BytesIO(raw_bytes), header=None)
                else:
                    # Has a header row
                    df_input = pd.read_csv(io.BytesIO(raw_bytes))

                # Extract true labels before passing to predictor (for display)
                has_label   = "label" in df_input.columns
                true_labels = df_input["label"].tolist() if has_label else None

                # Pass the raw DataFrame directly — NIDSPredictor handles all formats
                df_features = df_input.copy()

                n_display = df_features.shape[1]
                st.write(f"📋 **Loaded {len(df_features)} records** | {n_display} columns detected")
                st.info(
                    "Format detected: **Raw NSL-KDD** (no header)" if first_cell.lstrip("-").replace(".","").isdigit()
                    else "Format detected: **Named CSV** (with header)"
                )
                st.dataframe(df_features.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner(f"Predicting on {len(df_features)} records..."):
                        raw = predictor.predict(df_features)

                    # Build results DataFrame — use only first 41 numeric cols for display
                    key = ("ensemble" if active_model == "both"
                           else f"{active_model}_predictions")
                    predictions = raw.get(key, raw.get("rf_predictions", []))

                    df_results = df_features.copy()
                    df_results["predicted_class"] = predictions
                    if has_label:
                        df_results["true_label"] = true_labels

                    # Summary metrics
                    st.subheader("📊 Prediction Summary")
                    counts = pd.Series(predictions).value_counts()
                    c1, c2 = st.columns([1, 2])
                    with c1, st.container(border=True):
                        st.dataframe(counts.rename("Count").to_frame(), use_container_width=True)
                    with c2, st.container(border=True):
                        fig = px.pie(
                            values=counts.values, names=counts.index,
                            color=counts.index,
                            color_discrete_map=CLASS_COLORS,
                            title="Predicted Class Distribution",
                            hole=0.35,
                        )
                        fig.update_layout(
                            height=300,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Full results
                    st.subheader("📋 Detailed Results")
                    with st.container(border=True):
                        st.dataframe(df_results, use_container_width=True, height=300)

                    # Download
                    csv_bytes = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_bytes,
                        file_name="nids_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.title("📊 Model Evaluation Metrics")

    sub1, sub2, sub3 = st.tabs(["Random Forest", "LSTM", "Model Comparison"])

    with sub1:
        st.subheader("🌲 Random Forest Results")
        col1, col2 = st.columns(2)
        img = load_plot("rf_confusion_matrix.png")
        if img:
            with col1, st.container(border=True):
                st.image(img, caption="Confusion Matrix", use_container_width=True)
        img = load_plot("rf_per_class_metrics.png")
        if img:
            with col2, st.container(border=True):
                st.image(img, caption="Per-Class Metrics", use_container_width=True)
        img = load_plot("rf_feature_importances.png")
        if img:
            with st.container(border=True):
                st.image(img, caption="Top Feature Importances", use_container_width=True)
        if not os.path.exists(os.path.join(PLOTS_DIR, "rf_confusion_matrix.png")):
            st.info("Plots not found. Run `python main.py` to train models and generate plots.")

    with sub2:
        st.subheader("🧠 LSTM Results")
        col1, col2 = st.columns(2)
        img = load_plot("lstm_confusion_matrix.png")
        if img:
            with col1, st.container(border=True):
                st.image(img, caption="Confusion Matrix", use_container_width=True)
        img = load_plot("lstm_per_class_metrics.png")
        if img:
            with col2, st.container(border=True):
                st.image(img, caption="Per-Class Metrics", use_container_width=True)
        img = load_plot("lstm_training_history.png")
        if img:
            with st.container(border=True):
                st.image(img, caption="Training History", use_container_width=True)

    with sub3:
        st.subheader("⚖️ Model Comparison")
        img = load_plot("model_comparison.png")
        if img:
            with st.container(border=True):
                st.image(img, caption="RF vs LSTM — All Metrics", use_container_width=True)
        img = load_plot("comparison_confusion_matrices.png")
        if img:
            with st.container(border=True):
                st.image(img, caption="Side-by-Side Confusion Matrices", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Project Overview
    **Design and Implementation of a Network Intrusion Detection System Using
    Random Forest and Deep Learning**

    This final-year project develops an intelligent NIDS that detects and classifies
    network intrusions using machine learning and deep learning.

    ---

    ## Dataset: NSL-KDD
    | Split      | Records  |
    |------------|----------|
    | Train      | 125,973  |
    | Test       | 22,544   |

    The NSL-KDD dataset solves key problems of the original KDD Cup 1999 dataset:
    no duplicate records in the train/test sets, and a reasonable number of records
    in each difficulty group.

    ---

    ## Attack Classes
    | Class  | Description                                   | Examples                            |
    |--------|-----------------------------------------------|-------------------------------------|
    | Normal | Legitimate network traffic                    | —                                   |
    | DoS    | Denial of Service                             | neptune, smurf, back                |
    | Probe  | Surveillance / port scanning                  | ipsweep, nmap, satan                |
    | R2L    | Remote-to-Local unauthorised access           | ftp_write, guess_passwd             |
    | U2R    | User-to-Root privilege escalation             | buffer_overflow, rootkit            |

    ---

    ## Methodology
    1. **Preprocessing** — label mapping, missing value handling, one-hot encoding,
       column alignment, StandardScaler normalisation
    2. **Feature Selection** — Random Forest importance ranking (top 40 features retained)
    3. **Random Forest** — `RandomizedSearchCV` over n_estimators/max_depth/max_features/
       class_weight/criterion, optimised for macro-F1, with minority-class oversampling
    4. **LSTM** — Bidirectional LSTM (256→128→64 units) with LayerNorm and Dropout, Focal
       Loss + label smoothing, cosine-annealing LR, SMOTE oversampling for R2L/U2R
    5. **Evaluation** — Accuracy, Precision, Recall, F1-Score (weighted + macro), Confusion Matrix

    ---

    ## Tech Stack
    `Python 3.10` · `scikit-learn` · `TensorFlow / Keras` · `pandas` · `NumPy` · `Streamlit` · `Plotly`
    """)
