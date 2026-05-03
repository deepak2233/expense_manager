"""
Evaluation — self-consistency check, learnability test, and gold-standard
evaluation with confusion matrix.
"""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from . import config
from .utils import get_logger

log = get_logger(__name__)


# ─── Self-consistency ────────────────────────────────────────────────────────────

def check_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that identical normalised remarks received the same label.
    Fix any inconsistencies via majority-vote.
    """
    grouped = (
        df.groupby("Remarks_clean")["Predicted_Type"]
        .nunique()
        .reset_index()
        .rename(columns={"Predicted_Type": "n_labels"})
    )
    bad = grouped[grouped["n_labels"] > 1]

    if len(bad) == 0:
        log.info("Self-consistency: ✓ all identical remarks have identical labels")
        return df

    log.warning(f"Self-consistency: {len(bad)} remarks with inconsistent labels")
    for _, row in bad.iterrows():
        subset = df[df["Remarks_clean"] == row["Remarks_clean"]]
        log.warning(
            f"  '{row['Remarks_clean'][:80]}' → "
            f"{subset['Predicted_Type'].unique().tolist()}"
        )

    # Fix via majority-vote
    for _, row in bad.iterrows():
        mask = df["Remarks_clean"] == row["Remarks_clean"]
        majority = df.loc[mask, "Predicted_Type"].mode()[0]
        df.loc[mask, "Predicted_Type"] = majority

    log.info("Fixed via majority-vote")
    return df


# ─── Learnability test ──────────────────────────────────────────────────────────

def learnability_test(df: pd.DataFrame) -> float:
    """
    Train TF-IDF + LinearSVC with 5-fold stratified CV on the model's own labels
    (excluding OTHER).  If macro-F1 < threshold, labels are noisy.

    Returns the mean macro-F1 score.
    """
    mask = df["Predicted_Type"].isin(config.CANDIDATE_LABELS)
    # Convert to plain lists — pandas 3.x uses ArrowStringArray by default,
    # which sklearn can't index with numpy integer arrays.
    X = list(df.loc[mask, "Remarks_clean"])
    y = list(df.loc[mask, "Predicted_Type"])

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True),
        LinearSVC(class_weight="balanced", max_iter=5000),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    mean_f1 = scores.mean()

    if mean_f1 < config.LEARNABILITY_THRESHOLD:
        log.warning(
            f"Learnability: ⚠ macro-F1 = {mean_f1:.3f} (< {config.LEARNABILITY_THRESHOLD}) "
            f"— labels may be noisy"
        )
    else:
        log.info(
            f"Learnability: ✓ macro-F1 = {mean_f1:.3f} (≥ {config.LEARNABILITY_THRESHOLD})"
        )

    return mean_f1


# ─── Gold-standard evaluation ───────────────────────────────────────────────────

def _gold_label(text: str) -> object:
    """Assign a gold label using high-precision keyword rules from config."""
    t = str(text).lower()
    for cls, patterns in config.GOLD_RULES.items():
        for pat in patterns:
            if re.search(pat, t):
                return cls
    return np.nan


def evaluate_gold_standard(df: pd.DataFrame, plot_path: str) -> dict:
    """
    Build a gold-standard evaluation set via keyword rules and compute metrics.

    Returns a dict with accuracy, macro_f1, and the classification report string.
    """
    df["Gold_Label"] = df["Remarks_clean"].apply(_gold_label)
    eval_df = df.dropna(subset=["Gold_Label"]).copy()
    n = len(eval_df)
    log.info(f"Gold-standard evaluation set: {n} rows")

    if n == 0:
        log.warning("No rows matched gold rules — skipping evaluation")
        return {"accuracy": 0, "macro_f1": 0, "report": "N/A"}

    y_true = eval_df["Gold_Label"]
    y_pred = eval_df["Predicted_Type"]
    all_labels = config.CANDIDATE_LABELS + ["OTHER"]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0
    )
    # Extract macro F1 from report
    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]

    log.info(f"Accuracy: {acc*100:.1f}%")
    log.info(f"Macro F1: {macro_f1:.3f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=all_labels, yticklabels=all_labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (Gold)")
    ax.set_title("Confusion Matrix — Gold-Standard Evaluation")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix → {plot_path}")

    return {"accuracy": acc, "macro_f1": macro_f1, "report": report}
