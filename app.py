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
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS — Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    st.image("https://img.icons8.com/fluency/96/firewall.png", width=80)
    st.title("🛡️ NIDS Dashboard")
    st.markdown("**Network Intrusion Detection System**")
    st.markdown("*Random Forest + LSTM | NSL-KDD*")
    st.divider()

    model_choice = st.selectbox(
        "Active Model",
        options=["Both (Ensemble)", "Random Forest", "LSTM"],
        index=0
    )
    model_map = {"Both (Ensemble)": "both", "Random Forest": "rf", "LSTM": "lstm"}
    active_model = model_map[model_choice]

    st.divider()
    st.caption("Final Year Project")
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
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown(
        "An intelligent system that classifies network traffic into **Normal** "
        "or one of four attack categories using **Random Forest** and **LSTM** "
        "deep learning models trained on the **NSL-KDD** benchmark dataset."
    )

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
                f"""<div style='background:{CLASS_COLORS[cls]}22;border-left:4px solid
                {CLASS_COLORS[cls]};padding:12px;border-radius:6px;'>
                <b>{icons[cls]} {cls}</b><br><small>{descriptions[cls]}</small></div>""",
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader("📈 NSL-KDD Class Distribution (Train Set)")

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
    fig.update_layout(showlegend=False, height=380)
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
            st.markdown(f"""
            <div style='background:{display_color}22;border:2px solid {display_color};
            padding:22px;border-radius:10px;text-align:center;margin:15px 0'>
            <h2 style='color:{display_color};margin:0'>{display_icon} {display_cls}</h2>
            <p style='margin:8px 0;font-size:18px'>
                Ensemble Confidence: <b>{confidence*100:.1f}%</b>
            </p>
            <p style='margin:0;font-size:14px;color:#555'>
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
                        f"<div style='background:{rc}22;border:2px solid {rc};"
                        f"padding:12px;border-radius:8px;text-align:center'>"
                        f"<b>🌲 Random Forest</b><br>"
                        f"<span style='font-size:20px;color:{rc}'><b>{rf_vote}</b></span>"
                        f"</div>", unsafe_allow_html=True
                    )
                if lstm_vote:
                    lc = CLASS_COLORS.get(lstm_vote, "#888")
                    vc2.markdown(
                        f"<div style='background:{lc}22;border:2px solid {lc};"
                        f"padding:12px;border-radius:8px;text-align:center'>"
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
                    with c1:
                        st.dataframe(counts.rename("Count").to_frame(), use_container_width=True)
                    with c2:
                        fig = px.pie(
                            values=counts.values, names=counts.index,
                            color=counts.index,
                            color_discrete_map=CLASS_COLORS,
                            title="Predicted Class Distribution",
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    # Full results
                    st.subheader("📋 Detailed Results")
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
            col1.image(img, caption="Confusion Matrix", use_container_width=True)
        img = load_plot("rf_per_class_metrics.png")
        if img:
            col2.image(img, caption="Per-Class Metrics", use_container_width=True)
        img = load_plot("rf_feature_importances.png")
        if img:
            st.image(img, caption="Top Feature Importances", use_container_width=True)
        if not os.path.exists(os.path.join(PLOTS_DIR, "rf_confusion_matrix.png")):
            st.info("Plots not found. Run `python main.py` to train models and generate plots.")

    with sub2:
        st.subheader("🧠 LSTM Results")
        col1, col2 = st.columns(2)
        img = load_plot("lstm_confusion_matrix.png")
        if img:
            col1.image(img, caption="Confusion Matrix", use_container_width=True)
        img = load_plot("lstm_per_class_metrics.png")
        if img:
            col2.image(img, caption="Per-Class Metrics", use_container_width=True)
        img = load_plot("lstm_training_history.png")
        if img:
            st.image(img, caption="Training History", use_container_width=True)

    with sub3:
        st.subheader("⚖️ Model Comparison")
        img = load_plot("model_comparison.png")
        if img:
            st.image(img, caption="RF vs LSTM — All Metrics", use_container_width=True)
        img = load_plot("comparison_confusion_matrices.png")
        if img:
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
git remote add origin https://github.com/everthingdeen/NIDS.git
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
    3. **Random Forest** — 200 trees, balanced class weights, `sqrt` feature sampling
    4. **LSTM** — 2-layer stacked LSTM (128→64 units), BatchNorm, Dropout, input shape
       `(samples, 1, features)`, trained with EarlyStopping + ReduceLROnPlateau
    5. **Evaluation** — Accuracy, Precision, Recall, F1-Score (weighted), Confusion Matrix

    ---

    ## Tech Stack
    `Python 3.10` · `scikit-learn` · `TensorFlow / Keras` · `pandas` · `NumPy` · `Streamlit` · `Plotly`
    """)
