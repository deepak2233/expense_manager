"""
Classifier — zero-shot classification wrapper with confidence-based
human review queue.
"""

import pandas as pd
from tqdm.auto import tqdm

from . import config
from .utils import get_logger

log = get_logger(__name__)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run zero-shot classification on rows not already labelled (i.e. not OTHER).

    - Uses the model from config.ZERO_SHOT_MODEL.
    - Rows below config.CONFIDENCE_THRESHOLD are flagged for human review.
    - Returns the same DataFrame with Predicted_Type and Confidence filled.
    """
    from transformers import pipeline

    log.info(f"Loading model: {config.ZERO_SHOT_MODEL}")
    classifier = pipeline(
        "zero-shot-classification",
        model=config.ZERO_SHOT_MODEL,
    )

    mask = df["Predicted_Type"].isna()
    texts = df.loc[mask, "Remarks_clean"].tolist()
    log.info(f"Classifying {len(texts)} rows...")

    labels, scores = [], []
    for text in tqdm(texts, desc="Zero-shot"):
        if len(text.strip()) < config.MIN_TEXT_LENGTH:
            labels.append("OTHER")
            scores.append(0.0)
            continue
        res = classifier(text, config.CANDIDATE_LABELS)
        labels.append(res["labels"][0])
        scores.append(res["scores"][0])

    df.loc[mask, "Predicted_Type"] = labels
    df.loc[mask, "Confidence"] = scores

    # Flag low-confidence rows
    df["needs_review"] = (
        df["Confidence"] < config.CONFIDENCE_THRESHOLD
    ) & (~df["is_journal"])
    n_review = df["needs_review"].sum()
    log.info(f"Low-confidence rows for human review: {n_review}")

    log.info("Label distribution:")
    for label, count in df["Predicted_Type"].value_counts().items():
        log.info(f"  {label}: {count}")

    return df


def export_review_queue(df: pd.DataFrame, path: str) -> None:
    """Export low-confidence rows to an Excel file for human review."""
    review = df[df["needs_review"]].copy()
    review = review[["Amount", "Remarks_raw", "Remarks_clean",
                      "Predicted_Type", "Confidence"]]
    review = review.sort_values("Confidence")
    review.to_excel(path, index=False)
    log.info(f"Human review queue ({len(review)} rows) → {path}")
