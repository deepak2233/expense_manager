"""
Preprocessing — text normalisation and journal-entry pre-filter.

All logic is config-driven: edit config.py to change patterns/thresholds.
"""

import re

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

log = get_logger(__name__)


# ─── Text normalisation ─────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """
    Strip vendor PO prefixes, invoice numbers, date strings, and other noise
    that confuse classifiers.  Keeps the semantic payload.

    All patterns are read from config.NOISE_PATTERNS so they can be tuned
    without touching this function.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    t = text.strip()
    for pattern, replacement, ignore_case in config.NOISE_PATTERNS:
        flags = re.IGNORECASE if ignore_case else 0
        t = re.sub(pattern, replacement, t, flags=flags)
    return t.strip()


def normalise_column(series: pd.Series) -> pd.Series:
    """Apply normalise_text to an entire column."""
    return series.fillna("").apply(normalise_text)


# ─── Journal-entry pre-filter ────────────────────────────────────────────────────

def build_journal_mask(remarks_clean: pd.Series) -> pd.Series:
    """
    Return a boolean mask: True for rows that are accounting journal entries
    (provisions, reclasses, CWIP transfers) and should be routed to OTHER.
    """
    journal_regex = "|".join(config.JOURNAL_PATTERNS)
    mask = remarks_clean.str.lower().str.contains(journal_regex, regex=True, na=False)
    # Also flag empty / very short remarks
    mask = mask | (remarks_clean.str.strip().str.len() < config.MIN_TEXT_LENGTH)
    n = mask.sum()
    log.info(f"Journal pre-filter: {n} rows routed to OTHER")
    return mask


# ─── Full preprocessing pipeline ────────────────────────────────────────────────

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline:
      1. Drop redundant columns (Year, Credit, Net).
      2. Normalise the Remarks text.
      3. Pre-filter journal entries into OTHER.

    Returns a working DataFrame with columns:
      Amount, Remarks_raw, Remarks_clean, is_journal, Predicted_Type
    """
    log.info(f"Input: {len(df_raw)} rows, {df_raw.shape[1]} columns")

    # 1. Drop redundant columns
    df = df_raw[["Debit", "Remarks"]].copy()
    df.rename(columns={"Debit": "Amount"}, inplace=True)
    df["Remarks_raw"] = df["Remarks"].fillna("")
    log.info("Dropped Year, Credit, Net (all constant/redundant)")

    # 2. Normalise text
    df["Remarks_clean"] = normalise_column(df["Remarks_raw"])
    log.info("Text normalisation complete")

    # 3. Pre-filter journal entries
    df["is_journal"] = build_journal_mask(df["Remarks_clean"])
    df["Predicted_Type"] = pd.Series([None] * len(df), dtype="object")
    df.loc[df["is_journal"], "Predicted_Type"] = "OTHER"
    df["Confidence"] = np.nan
    df.loc[df["is_journal"], "Confidence"] = 1.0

    remaining = df["Predicted_Type"].isna().sum()
    log.info(f"Remaining for ML classification: {remaining}")

    return df
