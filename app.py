#!/usr/bin/env python3
"""
Streamlit Dashboard — Expense Classification Pipeline
Clean white professional UI.
"""

import sys, os, io, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

from src import config
from src.preprocessing import preprocess
from src.classifier import classify, export_review_queue
from src.evaluation import check_consistency, learnability_test, evaluate_gold_standard
from src.utils import ensure_dirs

# ─── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expense Classifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Clean white CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main .block-container { padding-top: 1.5rem; max-width: 1100px; }

/* Clean metric cards */
div[data-testid="stMetric"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label {
    color: #64748B;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1E293B;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #F8FAFC; }

h1, h2, h3 { color: #1E293B; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ───────────────────────────────────────────────────────────────
for key in ["df_raw", "df_result", "pipeline_run", "eval_results", "learn_f1"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Expense Classifier")
    st.caption("AI-powered expense categorisation")
    st.divider()

    source = st.radio("Data source", ["Upload file", "Use default data"])
    if source == "Upload file":
        uploaded = st.file_uploader("Upload .xlsx", type=["xlsx"])
        if uploaded:
            st.session_state.df_raw = pd.read_excel(uploaded)
            st.success(f"✓ {len(st.session_state.df_raw)} rows loaded")
    else:
        if os.path.exists(config.INPUT_FILE):
            st.session_state.df_raw = pd.read_excel(config.INPUT_FILE)
            st.success(f"✓ {len(st.session_state.df_raw)} rows loaded")
        else:
            st.error("Default data.xlsx not found")

    st.divider()
    st.markdown("**Settings**")
    conf_thresh = st.slider("Confidence threshold", 0.3, 0.9, config.CONFIDENCE_THRESHOLD, 0.05)
    st.divider()
    with st.expander("📖 Class Definitions"):
        for cls, desc in config.TAXONOMY.items():
            st.markdown(f"**{cls}:** {desc}")

# ─── Main area ───────────────────────────────────────────────────────────────────
st.title("Expense Classification Pipeline")
st.caption("Zero-shot NLP classification · Journal pre-filter · Human review queue")

if st.session_state.df_raw is None:
    st.info("← Select a data source from the sidebar to begin.")
    st.stop()

df_raw = st.session_state.df_raw
COLORS = {"Services": "#4F46E5", "Equipment": "#059669", "Material": "#D97706", "OTHER": "#DC2626"}

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab_eda, tab_run, tab_results, tab_review, tab_export = st.tabs([
    "📊 EDA", "▶️ Run Pipeline", "📈 Results", "🔍 Review Queue", "📥 Export"
])

# ═══ TAB 1: EDA ══════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Exploratory Data Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df_raw))
    c2.metric("Unique Remarks", df_raw["Remarks"].dropna().nunique() if "Remarks" in df_raw.columns else 0)
    c3.metric("Missing", int(df_raw["Remarks"].isna().sum()) if "Remarks" in df_raw.columns else 0)
    if "Remarks" in df_raw.columns:
        jk = ['provision', 'reclass', 'transfer', 'trf to', 'cwip', 'space matrix cost']
        c4.metric("Journal Entries", int(df_raw["Remarks"].fillna("").str.lower().apply(
            lambda t: any(p in t for p in jk)).sum()))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        amt = "Net" if "Net" in df_raw.columns else "Debit"
        if amt in df_raw.columns:
            fig = px.histogram(df_raw, x=amt, nbins=35, title="Expense Amount Distribution",
                               color_discrete_sequence=["#4F46E5"], opacity=0.8)
            fig.update_layout(height=320, margin=dict(t=40, b=20, l=40, r=20),
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Remarks" in df_raw.columns:
            fig2 = px.histogram(df_raw["Remarks"].dropna().str.len(), nbins=35,
                                title="Remark Length Distribution",
                                color_discrete_sequence=["#059669"], opacity=0.8,
                                labels={"value": "Characters"})
            fig2.update_layout(height=320, margin=dict(t=40, b=20, l=40, r=20), showlegend=False,
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig2, use_container_width=True)

    # Word cloud
    if "Remarks" in df_raw.columns:
        st.subheader("Word Cloud")
        text = re.sub(r"[^a-z\s]", "", " ".join(df_raw["Remarks"].dropna().str.lower()))
        if text.strip():
            wc = WordCloud(width=800, height=280, background_color="white",
                           colormap="Set2", max_words=100, min_word_length=3).generate(text)
            st.image(wc.to_array(), use_container_width=True)

    # Column check
    st.subheader("Column Audit")
    findings = []
    for col in df_raw.columns:
        if df_raw[col].nunique() == 1:
            findings.append(f"**{col}** = `{df_raw[col].iloc[0]}` (constant → droppable)")
    if "Net" in df_raw.columns and "Debit" in df_raw.columns and (df_raw["Net"] == df_raw["Debit"]).all():
        findings.append("**Net ≡ Debit** (redundant)")
    for f in findings:
        st.markdown(f"- {f}")
    if not findings:
        st.success("No redundant columns.")

    st.subheader("Data Preview")
    st.dataframe(df_raw, use_container_width=True, height=250)

# ═══ TAB 2: RUN ══════════════════════════════════════════════════════════════════
with tab_run:
    st.header("Run Classification Pipeline")

    col1, col2, col3 = st.columns(3)
    col1.markdown("**1. Preprocess**\n\nNormalise text, filter journals")
    col2.markdown("**2. Classify**\n\nDistilBERT zero-shot NLI")
    col3.markdown("**3. Validate**\n\nConsistency + learnability")

    st.markdown("---")

    if st.button("▶️  Run Pipeline", type="primary", use_container_width=True):
        config.CONFIDENCE_THRESHOLD = conf_thresh
        ensure_dirs(config.OUTPUT_DIR, config.PLOT_DIR)

        progress = st.progress(0, text="Starting...")

        progress.progress(10, text="Preprocessing...")
        df = preprocess(df_raw)

        progress.progress(20, text="Classifying (≈45s)...")
        df = classify(df)
        progress.progress(70, text="Checking consistency...")

        df = check_consistency(df)
        progress.progress(80, text="Learnability test...")

        learn_f1 = learnability_test(df)
        st.session_state.learn_f1 = learn_f1
        progress.progress(90, text="Evaluating...")

        cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
        eval_results = evaluate_gold_standard(df, cm_path)
        st.session_state.eval_results = eval_results
        st.session_state.df_result = df
        st.session_state.pipeline_run = True

        # Save files
        out = df_raw.copy()
        out["Predicted_Type"] = df["Predicted_Type"].values
        out["Confidence"] = df["Confidence"].values
        out["Remarks_Normalised"] = df["Remarks_clean"].values
        out["Needs_Human_Review"] = df["needs_review"].values
        out.to_excel(config.OUTPUT_CLASSIFIED, index=False)
        export_review_queue(df, config.OUTPUT_REVIEW_QUEUE)

        progress.progress(100, text="✅ Done!")
        time.sleep(1)
        st.rerun()

    if st.session_state.pipeline_run:
        st.success("Pipeline complete. See **Results** and **Review Queue** tabs.")

# ═══ TAB 3: RESULTS ══════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first.")
        st.stop()

    df = st.session_state.df_result
    ev = st.session_state.eval_results
    lf = st.session_state.learn_f1

    st.header("Classification Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{ev['accuracy']*100:.1f}%")
    c2.metric("Macro F1", f"{ev['macro_f1']:.3f}")
    c3.metric("Learnability", f"{lf:.3f}",
              delta="Clean" if lf >= config.LEARNABILITY_THRESHOLD else "⚠ Noisy",
              delta_color="normal" if lf >= config.LEARNABILITY_THRESHOLD else "inverse")
    c4.metric("Needs Review", int(df["needs_review"].sum()))

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        counts = df["Predicted_Type"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        fig = px.pie(counts, names="Class", values="Count", hole=0.45,
                     title="Classification Distribution",
                     color="Class", color_discrete_map=COLORS)
        fig.update_traces(textinfo="label+value", textfont_size=12)
        fig.update_layout(height=380, margin=dict(t=50, b=10),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        mask_main = df["Predicted_Type"].isin(config.CANDIDATE_LABELS)
        if mask_main.any():
            fig2 = px.histogram(df[mask_main], x="Confidence", color="Predicted_Type",
                                nbins=20, title="Confidence Scores",
                                color_discrete_map=COLORS, barmode="overlay", opacity=0.7)
            fig2.add_vline(x=config.CONFIDENCE_THRESHOLD, line_dash="dash", line_color="#DC2626",
                           annotation_text="Threshold")
            fig2.update_layout(height=380, margin=dict(t=50, b=10),
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig2, use_container_width=True)

    # Spend by class
    st.subheader("Spend by Category")
    spend = df.groupby("Predicted_Type")["Amount"].sum().sort_values(ascending=True)
    fig3 = px.bar(y=spend.index, x=spend.values, orientation="h",
                  color=spend.index, color_discrete_map=COLORS,
                  labels={"x": "Total Amount", "y": ""})
    fig3.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False,
                       plot_bgcolor="white", paper_bgcolor="white",
                       xaxis=dict(gridcolor="#F1F5F9"))
    st.plotly_chart(fig3, use_container_width=True)

    # Confusion matrix
    cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.subheader("Confusion Matrix")
        st.image(cm_path, width=550)

    # Data table
    st.subheader("Classified Data")
    show_cols = [c for c in ["Amount", "Remarks_raw", "Predicted_Type", "Confidence"] if c in df.columns]
    filter_cls = st.multiselect("Filter:", config.CANDIDATE_LABELS + ["OTHER"],
                                default=config.CANDIDATE_LABELS + ["OTHER"])
    st.dataframe(df[df["Predicted_Type"].isin(filter_cls)][show_cols],
                 use_container_width=True, height=350)

# ═══ TAB 4: REVIEW ═══════════════════════════════════════════════════════════════
with tab_review:
    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first.")
        st.stop()

    df = st.session_state.df_result
    review = df[df["needs_review"]].sort_values("Confidence")

    st.header("Human Review Queue")
    st.markdown(f"**{len(review)} rows** below **{config.CONFIDENCE_THRESHOLD:.0%}** confidence.")

    if len(review) == 0:
        st.success("All predictions are above the confidence threshold.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 < 40%", int((review["Confidence"] < 0.40).sum()))
        c2.metric("🟡 40–50%", int(((review["Confidence"] >= 0.40) & (review["Confidence"] < 0.50)).sum()))
        c3.metric("🟢 50–55%", int((review["Confidence"] >= 0.50).sum()))

        st.markdown("---")
        show = [c for c in ["Amount", "Remarks_raw", "Predicted_Type", "Confidence"] if c in review.columns]
        st.dataframe(review[show], use_container_width=True, height=450)

# ═══ TAB 5: EXPORT ═══════════════════════════════════════════════════════════════
with tab_export:
    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first.")
        st.stop()

    df = st.session_state.df_result
    st.header("Download Outputs")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Classified Data")
        buf = io.BytesIO()
        out = df_raw.copy()
        for col_name, src_col in [("Predicted_Type", "Predicted_Type"), ("Confidence", "Confidence"),
                                   ("Remarks_Normalised", "Remarks_clean"), ("Needs_Review", "needs_review")]:
            out[col_name] = df[src_col].values
        out.to_excel(buf, index=False, engine="openpyxl")
        st.download_button("⬇️  classified_data.xlsx", buf.getvalue(), "classified_data.xlsx",
                           use_container_width=True)

    with col2:
        st.subheader("🔍 Review Queue")
        buf2 = io.BytesIO()
        review = df[df["needs_review"]].sort_values("Confidence")
        rcols = [c for c in ["Amount", "Remarks_raw", "Remarks_clean", "Predicted_Type", "Confidence"] if c in review.columns]
        review[rcols].to_excel(buf2, index=False, engine="openpyxl")
        st.download_button("⬇️  review_queue.xlsx", buf2.getvalue(), "review_queue.xlsx",
                           use_container_width=True)

    st.markdown("---")
    st.subheader("Summary")
    summary = df["Predicted_Type"].value_counts().reset_index()
    summary.columns = ["Class", "Count"]
    summary["Percentage"] = (summary["Count"] / summary["Count"].sum() * 100).round(1)
    st.dataframe(summary, use_container_width=True, hide_index=True)
