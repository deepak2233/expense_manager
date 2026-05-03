#!/usr/bin/env python3
"""
Streamlit Dashboard — Expense Classification Pipeline
Premium UI with glassmorphism, dark theme, interactive charts.
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
from src.preprocessing import preprocess, normalise_text
from src.classifier import classify, export_review_queue
from src.evaluation import check_consistency, learnability_test, evaluate_gold_standard
from src.utils import ensure_dirs

# ─── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expense Classifier — AI Pipeline",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

div[data-testid="stMetric"] label {
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    opacity: 0.8;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700;
    font-size: 1.8rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    opacity: 0.65;
    margin-bottom: 2rem;
}

.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.pill-green { background: rgba(34,197,94,0.15); color: #22c55e; }
.pill-amber { background: rgba(245,158,11,0.15); color: #f59e0b; }
.pill-red   { background: rgba(239,68,68,0.15); color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ───────────────────────────────────────────────────────────────
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_result" not in st.session_state:
    st.session_state.df_result = None
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "learn_f1" not in st.session_state:
    st.session_state.learn_f1 = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏷️ Expense Classifier")
    st.markdown("AI-powered expense categorisation pipeline")
    st.divider()

    st.markdown("### 📁 Data Source")
    source = st.radio("Choose input:", ["Upload file", "Use default data"], label_visibility="collapsed")

    if source == "Upload file":
        uploaded = st.file_uploader("Upload .xlsx file", type=["xlsx"])
        if uploaded:
            st.session_state.df_raw = pd.read_excel(uploaded)
            st.success(f"✓ Loaded {len(st.session_state.df_raw)} rows")
    else:
        if os.path.exists(config.INPUT_FILE):
            st.session_state.df_raw = pd.read_excel(config.INPUT_FILE)
            st.success(f"✓ Default data: {len(st.session_state.df_raw)} rows")
        else:
            st.error("Default data.xlsx not found")

    st.divider()
    st.markdown("### ⚙️ Configuration")
    conf_thresh = st.slider("Confidence threshold", 0.3, 0.9, config.CONFIDENCE_THRESHOLD, 0.05,
                            help="Predictions below this are sent to human review")
    learn_thresh = st.slider("Learnability threshold", 0.5, 0.95, config.LEARNABILITY_THRESHOLD, 0.05,
                             help="CV F1 below this triggers a noisy-label alarm")

    st.divider()
    st.markdown("### 📖 Taxonomy")
    with st.expander("View class definitions"):
        for cls, desc in config.TAXONOMY.items():
            st.markdown(f"**{cls}:** {desc}")

# ─── Hero header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">Expense Classification Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Zero-shot NLP classification · Pre-filtering · Human review queue · Self-diagnostics</p>', unsafe_allow_html=True)

if st.session_state.df_raw is None:
    st.info("👈 Upload a dataset or select default data from the sidebar to begin.")
    st.stop()

df_raw = st.session_state.df_raw

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab_eda, tab_run, tab_results, tab_review, tab_export = st.tabs([
    "📊 EDA", "🚀 Run Pipeline", "📈 Results", "🔍 Human Review", "📥 Export"
])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1: EDA
# ══════════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("## Exploratory Data Analysis")
    st.markdown("Findings that drive pipeline design decisions.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df_raw))
    n_unique = df_raw["Remarks"].dropna().nunique() if "Remarks" in df_raw.columns else 0
    col2.metric("Unique Remarks", n_unique)
    col3.metric("Missing Remarks", int(df_raw["Remarks"].isna().sum()) if "Remarks" in df_raw.columns else 0)

    # Detect journal entries
    if "Remarks" in df_raw.columns:
        jk = ['provision', 'reclass', 'transfer', 'trf to', 'cwip', 'space matrix cost']
        jmask = df_raw["Remarks"].fillna("").str.lower().apply(lambda t: any(p in t for p in jk))
        col4.metric("Journal Entries", int(jmask.sum()))

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if "Debit" in df_raw.columns or "Net" in df_raw.columns:
            amount_col = "Net" if "Net" in df_raw.columns else "Debit"
            fig = px.histogram(df_raw, x=amount_col, nbins=40, title="Distribution of Expense Amounts",
                               color_discrete_sequence=["#6366f1"], template="plotly_dark",
                               opacity=0.85)
            fig.update_layout(bargap=0.05, height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "Remarks" in df_raw.columns:
            lengths = df_raw["Remarks"].dropna().str.len()
            fig2 = px.histogram(lengths, nbins=40, title="Distribution of Remark Lengths",
                                color_discrete_sequence=["#22c55e"], template="plotly_dark",
                                opacity=0.85, labels={"value": "Characters"})
            fig2.update_layout(bargap=0.05, height=350, margin=dict(t=40, b=30), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Word cloud
    if "Remarks" in df_raw.columns:
        st.markdown("### 🔤 Word Cloud of Expense Remarks")
        all_text = " ".join(df_raw["Remarks"].dropna().str.lower().tolist())
        clean_text = re.sub(r"[^a-z\s]", "", all_text)
        if clean_text.strip():
            wc = WordCloud(width=900, height=350, background_color="#0e1117",
                           colormap="cool", max_words=120, min_word_length=3).generate(clean_text)
            st.image(wc.to_array(), use_container_width=True)

    # Redundancy findings
    st.markdown("### 🔍 Column Redundancy Check")
    findings = []
    for col in df_raw.columns:
        nuniq = df_raw[col].nunique()
        if nuniq == 1:
            findings.append(f"**{col}** — constant (`{df_raw[col].iloc[0]}`), can be dropped")
    if "Net" in df_raw.columns and "Debit" in df_raw.columns:
        if (df_raw["Net"] == df_raw["Debit"]).all():
            findings.append("**Net ≡ Debit** — redundant column")
    if findings:
        for f in findings:
            st.markdown(f"- {f}")
    else:
        st.success("No redundant columns detected.")

    st.divider()
    st.markdown("### 📋 Raw Data Preview")
    st.dataframe(df_raw, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2: Run Pipeline
# ══════════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.markdown("## 🚀 Run Classification Pipeline")
    st.markdown("Execute the full end-to-end pipeline: preprocess → classify → validate.")

    steps = [
        ("1️⃣", "Text Normalisation", "Strip PO numbers, dates, invoice codes"),
        ("2️⃣", "Journal Pre-filter", "Route accounting entries to OTHER"),
        ("3️⃣", "Zero-Shot Classification", "DistilBERT-MNLI on cleaned text"),
        ("4️⃣", "Self-Consistency Check", "Same text → same label?"),
        ("5️⃣", "Learnability Alarm", "TF-IDF + SVC 5-fold CV"),
        ("6️⃣", "Gold-Standard Evaluation", "Keyword-based ground truth"),
    ]

    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"**{icon} {name}**")
            st.caption(desc)

    st.divider()

    if st.button("▶️  Run Full Pipeline", type="primary", use_container_width=True):
        config.CONFIDENCE_THRESHOLD = conf_thresh
        config.LEARNABILITY_THRESHOLD = learn_thresh
        ensure_dirs(config.OUTPUT_DIR, config.PLOT_DIR)

        progress = st.progress(0, text="Starting pipeline...")
        status_area = st.empty()

        # Step 1-2: Preprocess
        status_area.info("🔄 Preprocessing: normalising text + filtering journal entries...")
        progress.progress(10, text="Preprocessing...")
        df = preprocess(df_raw)
        progress.progress(20, text="Preprocessing complete")

        # Step 3: Classify
        status_area.info("🤖 Running zero-shot classification (this may take ~45s)...")
        progress.progress(25, text="Classifying...")
        df = classify(df)
        progress.progress(70, text="Classification complete")

        # Step 4: Consistency
        status_area.info("🔍 Checking self-consistency...")
        df = check_consistency(df)
        progress.progress(80, text="Consistency check done")

        # Step 5: Learnability
        status_area.info("📐 Running learnability test...")
        learn_f1 = learnability_test(df)
        st.session_state.learn_f1 = learn_f1
        progress.progress(90, text="Learnability check done")

        # Step 6: Evaluation
        status_area.info("📊 Evaluating against gold standard...")
        cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
        eval_results = evaluate_gold_standard(df, cm_path)
        st.session_state.eval_results = eval_results
        progress.progress(100, text="✅ Pipeline complete!")

        # Store results
        st.session_state.df_result = df
        st.session_state.pipeline_run = True

        # Export files
        out = df_raw.copy()
        out["Predicted_Type"] = df["Predicted_Type"].values
        out["Confidence"] = df["Confidence"].values
        out["Remarks_Normalised"] = df["Remarks_clean"].values
        out["Needs_Human_Review"] = df["needs_review"].values
        out.to_excel(config.OUTPUT_CLASSIFIED, index=False)
        export_review_queue(df, config.OUTPUT_REVIEW_QUEUE)

        status_area.success("✅ Pipeline finished successfully!")
        time.sleep(1)
        st.rerun()

    if st.session_state.pipeline_run:
        st.success("✅ Pipeline has been run. Check the **Results** and **Human Review** tabs.")

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3: Results
# ══════════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    df = st.session_state.df_result
    eval_r = st.session_state.eval_results
    learn_f1 = st.session_state.learn_f1

    st.markdown("## 📈 Classification Results")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{eval_r['accuracy']*100:.1f}%")
    c2.metric("Macro F1", f"{eval_r['macro_f1']:.3f}")

    learn_delta = "✓ Clean" if learn_f1 >= config.LEARNABILITY_THRESHOLD else "⚠ Noisy"
    c3.metric("Learnability F1", f"{learn_f1:.3f}", delta=learn_delta,
              delta_color="normal" if learn_f1 >= config.LEARNABILITY_THRESHOLD else "inverse")
    c4.metric("Review Queue", f"{df['needs_review'].sum()}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        # Distribution donut
        counts = df["Predicted_Type"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        colors = {"Services": "#6366f1", "Equipment": "#22c55e", "Material": "#f59e0b", "OTHER": "#ef4444"}
        fig = px.pie(counts, names="Class", values="Count", hole=0.5,
                     title="Classification Distribution",
                     color="Class", color_discrete_map=colors,
                     template="plotly_dark")
        fig.update_traces(textinfo="label+value", textfont_size=13)
        fig.update_layout(height=400, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Confidence distribution
        mask_main = df["Predicted_Type"].isin(config.CANDIDATE_LABELS)
        if mask_main.any():
            fig2 = px.histogram(df[mask_main], x="Confidence", color="Predicted_Type",
                                nbins=25, title="Confidence Score Distribution",
                                color_discrete_map=colors, template="plotly_dark",
                                barmode="overlay", opacity=0.7)
            fig2.add_vline(x=config.CONFIDENCE_THRESHOLD, line_dash="dash",
                           line_color="red", annotation_text="Review threshold")
            fig2.update_layout(height=400, margin=dict(t=50, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrix
    cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.markdown("### 🗺️ Confusion Matrix (Gold-Standard)")
        st.image(cm_path, width=600)

    # Amount by class
    st.markdown("### 💰 Total Expenditure by Class")
    amount_by_class = df.groupby("Predicted_Type")["Amount"].sum().sort_values(ascending=False)
    fig3 = px.bar(x=amount_by_class.index, y=amount_by_class.values,
                  title="Total Spend by Category",
                  labels={"x": "Class", "y": "Total Amount"},
                  color=amount_by_class.index, color_discrete_map=colors,
                  template="plotly_dark")
    fig3.update_layout(height=350, margin=dict(t=50, b=30), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Full classified table
    st.markdown("### 📋 Full Classified Data")
    display_cols = ["Amount", "Remarks_raw", "Remarks_clean", "Predicted_Type", "Confidence"]
    avail_cols = [c for c in display_cols if c in df.columns]
    filter_class = st.multiselect("Filter by class:", config.CANDIDATE_LABELS + ["OTHER"],
                                  default=config.CANDIDATE_LABELS + ["OTHER"])
    filtered = df[df["Predicted_Type"].isin(filter_class)]
    st.dataframe(filtered[avail_cols], use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 4: Human Review
# ══════════════════════════════════════════════════════════════════════════════════
with tab_review:
    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first to see the review queue.")
        st.stop()

    df = st.session_state.df_result
    review_df = df[df["needs_review"]].copy().sort_values("Confidence")

    st.markdown("## 🔍 Human Review Queue")
    st.markdown(f"**{len(review_df)} rows** with confidence below **{config.CONFIDENCE_THRESHOLD:.0%}** need expert review.")

    if len(review_df) == 0:
        st.success("No rows need review — all predictions are above the confidence threshold.")
    else:
        # Confidence buckets
        st.markdown("### Confidence Breakdown")
        c1, c2, c3 = st.columns(3)
        low = (review_df["Confidence"] < 0.40).sum()
        mid = ((review_df["Confidence"] >= 0.40) & (review_df["Confidence"] < 0.50)).sum()
        high = (review_df["Confidence"] >= 0.50).sum()
        c1.metric("🔴 Very Low (<40%)", low)
        c2.metric("🟡 Low (40-50%)", mid)
        c3.metric("🟢 Borderline (50-55%)", high)

        st.divider()

        display_cols = ["Amount", "Remarks_raw", "Predicted_Type", "Confidence"]
        avail = [c for c in display_cols if c in review_df.columns]
        st.dataframe(
            review_df[avail],
            use_container_width=True, height=500
        )

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 5: Export
# ══════════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("## 📥 Download Outputs")

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first to generate downloadable outputs.")
        st.stop()

    df = st.session_state.df_result
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📊 Classified Data")
        st.caption("All rows with predicted types, confidence scores, and review flags.")
        buf1 = io.BytesIO()
        out = df_raw.copy()
        out["Predicted_Type"] = df["Predicted_Type"].values
        out["Confidence"] = df["Confidence"].values
        out["Remarks_Normalised"] = df["Remarks_clean"].values
        out["Needs_Human_Review"] = df["needs_review"].values
        out.to_excel(buf1, index=False, engine="openpyxl")
        st.download_button("⬇️  Download classified_data.xlsx", buf1.getvalue(),
                           "classified_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    with c2:
        st.markdown("### 🔍 Human Review Queue")
        st.caption("Low-confidence predictions sorted for expert review.")
        review_df = df[df["needs_review"]].copy().sort_values("Confidence")
        buf2 = io.BytesIO()
        review_out = review_df[["Amount", "Remarks_raw", "Remarks_clean", "Predicted_Type", "Confidence"]]
        review_out.to_excel(buf2, index=False, engine="openpyxl")
        st.download_button("⬇️  Download human_review_queue.xlsx", buf2.getvalue(),
                           "human_review_queue.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    # Summary stats download as CSV
    st.divider()
    st.markdown("### 📈 Summary Statistics")
    summary = df["Predicted_Type"].value_counts().reset_index()
    summary.columns = ["Class", "Count"]
    summary["Percentage"] = (summary["Count"] / summary["Count"].sum() * 100).round(1)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    csv = summary.to_csv(index=False)
    st.download_button("⬇️  Download summary.csv", csv, "summary.csv", "text/csv", use_container_width=True)
