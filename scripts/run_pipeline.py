#!/usr/bin/env python3
"""
run_pipeline.py — One-command end-to-end expense classification pipeline.

Usage:
    python scripts/run_pipeline.py

Reads data/raw/data.xlsx → runs full pipeline → writes outputs/.
"""

import sys
import os
import time

# Ensure user-local packages are importable
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.12/site-packages"))
# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.utils import get_logger, ensure_dirs
from src.preprocessing import preprocess
from src.classifier import classify, export_review_queue
from src.evaluation import check_consistency, learnability_test, evaluate_gold_standard


log = get_logger("pipeline")


def run_eda(df_raw: pd.DataFrame) -> None:
    """Produce EDA plots that drove pipeline design decisions."""
    log.info("── EDA ──")
    log.info(f"Year unique: {df_raw['Year'].unique().tolist()}")
    log.info(f"Credit unique: {df_raw['Credit'].unique().tolist()}")
    log.info(f"Net ≡ Debit: {(df_raw['Net'] == df_raw['Debit']).all()}")
    log.info(f"Missing Remarks: {df_raw['Remarks'].isna().sum()}")
    n_unique = df_raw["Remarks"].dropna().nunique()
    log.info(f"Unique remarks: {n_unique} / {len(df_raw)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df_raw["Net"], bins=40, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution of Net Expense")
    sns.histplot(
        df_raw["Remarks"].dropna().str.len(), bins=40, kde=True,
        ax=axes[1], color="seagreen",
    )
    axes[1].set_title("Distribution of Remark Length (chars)")
    plt.tight_layout()
    path = os.path.join(config.PLOT_DIR, "eda_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"EDA plot → {path}")


def save_final(df_raw: pd.DataFrame, df: pd.DataFrame) -> None:
    """Save the classified output and final distribution plot."""
    out = df_raw.copy()
    out["Predicted_Type"] = df["Predicted_Type"].values
    out["Confidence"] = df["Confidence"].values
    out["Remarks_Normalised"] = df["Remarks_clean"].values
    out["Needs_Human_Review"] = df["needs_review"].values
    out.to_excel(config.OUTPUT_CLASSIFIED, index=False)
    log.info(f"Classified data → {config.OUTPUT_CLASSIFIED}")

    # Distribution plot
    fig, ax = plt.subplots(figsize=(8, 5))
    order = config.CANDIDATE_LABELS + ["OTHER"]
    sns.countplot(data=df, x="Predicted_Type", order=order, palette="Set2", ax=ax)
    ax.set_title("Final Expense Categorisation (All Rows)")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    plt.tight_layout()
    path = os.path.join(config.PLOT_DIR, "final_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Distribution plot → {path}")


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("EXPENSE CLASSIFICATION PIPELINE — START")
    log.info("=" * 60)

    # Ensure output dirs exist
    ensure_dirs(config.OUTPUT_DIR, config.PLOT_DIR)

    # ── 1. Load raw data ─────────────────────────────────────────────────────
    log.info(f"Loading {config.INPUT_FILE}")
    df_raw = pd.read_excel(config.INPUT_FILE)
    log.info(f"Loaded {len(df_raw)} rows")

    # ── 2. EDA ───────────────────────────────────────────────────────────────
    run_eda(df_raw)

    # ── 3. Preprocess ────────────────────────────────────────────────────────
    df = preprocess(df_raw)

    # ── 4. Classify ──────────────────────────────────────────────────────────
    df = classify(df)

    # ── 5. Self-consistency ──────────────────────────────────────────────────
    df = check_consistency(df)

    # ── 6. Learnability alarm ────────────────────────────────────────────────
    learnability_test(df)

    # ── 7. Gold-standard evaluation ──────────────────────────────────────────
    cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    results = evaluate_gold_standard(df, cm_path)
    log.info(f"Gold-standard accuracy: {results['accuracy']*100:.1f}%")
    log.info(f"Gold-standard macro-F1: {results['macro_f1']:.3f}")

    # ── 8. Save outputs ─────────────────────────────────────────────────────
    save_final(df_raw, df)
    export_review_queue(df, config.OUTPUT_REVIEW_QUEUE)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
