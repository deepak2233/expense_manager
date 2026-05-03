#!/usr/bin/env python3
"""
build_notebook.py — Generates the Jupyter notebook from the modular pipeline.

Usage:
    python scripts/build_notebook.py
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.12/site-packages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nbformat as nbf
from src import config

nb = nbf.v4.new_notebook()

# ─── Taxonomy spec ───────────────────────────────────────────────────────────────
md_title = """\
# Expense Classification Pipeline — AI/ML/NLP Candidate Test

## Taxonomy Specification (written *before* any modelling)

Before writing a single line of model code we need a clear, reviewer-testable
definition of each class.  If two annotators can't agree on a row, the class
definition is too loose.

| Class | Definition | Prototypical items |
|-------|------------|--------------------|
| **Equipment** | A discrete, durable physical asset purchased as a single unit and typically inventoried (useful life > 1 year). | AC unit, dispenser, almirah, TV, projector, UPS, camera, chair, desk, kiosk |
| **Material** | A consumable, commodity, or bulk-purchased input that is installed/used as part of a larger system and is not individually tracked as an asset. | Copper pipe, PVC pipe, cable, wire, insulation, valves, conduit, nut-bolts, clamps |
| **Services** | A labour-intensive engagement — professional work, installation, testing, commissioning, consultancy, or construction/civil/MEP work billed as a service. | Consultancy fees, installation charges, civil work, carpentry work, fire-fighting system commissioning, interior/MEP work |
| **OTHER** | Accounting journal entries, provisions, reclassifications, transfers, or any remark that does not describe a procured good or service. Must **not** be force-fit into the above three. | "Provision reclass of Godrej May22", "Trf to Exp to Cwip", "Exp reclass entry", "Expense Provision for Sep22" |

> **Design decision:** We always include an out-of-scope class. A production classifier
> that silently force-fits journal entries into Material corrupts the books.
"""

# ─── Imports ─────────────────────────────────────────────────────────────────────
code_imports = """\
import sys, os, re, warnings
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.12/site-packages'))
sys.path.insert(0, '..')       # so 'from src import ...' works from notebooks/
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.preprocessing import preprocess, normalise_text
from src.classifier import classify, export_review_queue
from src.evaluation import check_consistency, learnability_test, evaluate_gold_standard
from src.utils import ensure_dirs

ensure_dirs(config.OUTPUT_DIR, config.PLOT_DIR)

df_raw = pd.read_excel(config.INPUT_FILE)
print(f"Raw dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
display(df_raw.head())
"""

# ─── EDA ─────────────────────────────────────────────────────────────────────────
md_eda = """\
## 1. EDA — finding things that change what we build

Good EDA should surface at least five actionable findings before modelling begins.
"""

code_eda = """\
# ── Finding 1: constant / redundant columns ──────────────────────────────────
print("Year unique values:", df_raw['Year'].unique().tolist())
print("Credit unique values:", df_raw['Credit'].unique().tolist())
print("Net ≡ Debit?", (df_raw['Net'] == df_raw['Debit']).all())
print("→ Year is constant (FY23), Credit ≡ 0, Net ≡ Debit.  All three are droppable.\\n")

# ── Finding 2: missing values ────────────────────────────────────────────────
print("Missing per column:")
print(df_raw.isnull().sum())
missing_idx = df_raw[df_raw['Remarks'].isna()].index[0]
missing_amt = df_raw.loc[missing_idx, 'Debit']
print(f"\\n→ 1 row has NaN Remarks (index {missing_idx}, Debit = {missing_amt}).  Routed to OTHER.\\n")

# ── Finding 3: unique-remarks audit ──────────────────────────────────────────
n_total, n_unique = len(df_raw), df_raw['Remarks'].dropna().nunique()
print(f"Total rows: {n_total}, Unique remark strings: {n_unique}")
print(f"→ {n_total - n_unique - 1} rows are exact duplicates of another remark.\\n")

# ── Finding 4: journal-entry rows ────────────────────────────────────────────
journal_keywords = ['provision', 'reclass', 'transfer', 'trf to', 'cwip', 'space matrix cost']
mask_j = df_raw['Remarks'].fillna('').str.lower().apply(lambda t: any(p in t for p in journal_keywords))
print(f"Rows matching journal-entry patterns: {mask_j.sum()}")
for r in df_raw.loc[mask_j, 'Remarks'].tolist():
    print(f"  • {r[:100]}")
print("→ These are accounting adjustments, NOT purchases.  Must go to OTHER.\\n")

# ── Finding 5: distributions ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df_raw['Net'], bins=40, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Distribution of Net Expense')
sns.histplot(df_raw['Remarks'].dropna().str.len(), bins=40, kde=True, ax=axes[1], color='seagreen')
axes[1].set_title('Distribution of Remark Length (chars)')
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, 'eda_distributions.png'), dpi=150)
plt.show()
"""

# ─── Preprocessing ───────────────────────────────────────────────────────────────
md_preproc = """\
## 2. Preprocessing — text normalisation + journal pre-filter

TF-IDF amplifies whatever you give it.  Vendor PO prefixes, invoice numbers,
and date strings are noise that can flip a classification.  We strip them first,
then deterministically route journal entries to OTHER.
"""

code_preproc = """\
df = preprocess(df_raw)

# Show before/after for noisy rows
for i in [0, 5, 7]:
    if i < len(df):
        print(f"[{i}] RAW:   {df.loc[i, 'Remarks_raw'][:100]}")
        print(f"     CLEAN: {df.loc[i, 'Remarks_clean'][:100]}")
        print()

print(f"\\nJournal entries routed to OTHER: {df['is_journal'].sum()}")
print(f"Remaining for ML: {df['Predicted_Type'].isna().sum()}")
"""

# ─── Classification ──────────────────────────────────────────────────────────────
md_classify = """\
## 3. Zero-Shot Classification (on cleaned text, 3 classes only)

Model: `typeform/distilbert-base-uncased-mnli`.  Journal entries have already
been removed, so the model only chooses among Services / Equipment / Material.
Rows with confidence < 0.55 are flagged for human review.
"""

code_classify = """\
df = classify(df)

# Export human review queue
export_review_queue(df, config.OUTPUT_REVIEW_QUEUE)

print(f"\\nRows needing human review: {df['needs_review'].sum()}")
print(f"\\nLabel distribution:")
print(df['Predicted_Type'].value_counts())
"""

# ─── Validation ──────────────────────────────────────────────────────────────────
md_validate = """\
## 4. Self-consistency & learnability checks

Two cheap diagnostics (~20 lines of sklearn) that catch noisy labels before deployment.
"""

code_validate = """\
# Self-consistency
df = check_consistency(df)

# Learnability
mean_f1 = learnability_test(df)
print(f"Learnability macro-F1: {mean_f1:.3f}")
"""

# ─── Gold-standard eval ─────────────────────────────────────────────────────────
md_eval = """\
## 5. Evaluation against hand-curated ground truth

High-precision keyword rules (used ONLY for evaluation — the model never saw them).
"""

code_eval = """\
cm_path = os.path.join(config.PLOT_DIR, 'confusion_matrix.png')
results = evaluate_gold_standard(df, cm_path)

print(f"\\nAccuracy: {results['accuracy']*100:.1f}%")
print(f"Macro F1: {results['macro_f1']:.3f}")
print(f"\\n{results['report']}")

# Show the confusion matrix image
from IPython.display import Image
display(Image(filename=cm_path))
"""

# ─── Final output ────────────────────────────────────────────────────────────────
md_final = """\
## 6. Final output and distribution
"""

code_final = """\
fig, ax = plt.subplots(figsize=(8, 5))
order = config.CANDIDATE_LABELS + ['OTHER']
sns.countplot(data=df, x='Predicted_Type', order=order, palette='Set2', ax=ax)
ax.set_title('Final Expense Categorisation (All Rows)')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, 'final_distribution.png'), dpi=150)
plt.show()

# Save final output
out = df_raw.copy()
out['Predicted_Type'] = df['Predicted_Type'].values
out['Confidence'] = df['Confidence'].values
out['Remarks_Normalised'] = df['Remarks_clean'].values
out['Needs_Human_Review'] = df['needs_review'].values
out.to_excel(config.OUTPUT_CLASSIFIED, index=False)
print(f"\\n→ Results saved to {config.OUTPUT_CLASSIFIED}")
print(f"→ Human review queue saved to {config.OUTPUT_REVIEW_QUEUE}")
"""

# ─── Assemble ────────────────────────────────────────────────────────────────────
nb.cells = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(md_eda),
    nbf.v4.new_code_cell(code_eda),
    nbf.v4.new_markdown_cell(md_preproc),
    nbf.v4.new_code_cell(code_preproc),
    nbf.v4.new_markdown_cell(md_classify),
    nbf.v4.new_code_cell(code_classify),
    nbf.v4.new_markdown_cell(md_validate),
    nbf.v4.new_code_cell(code_validate),
    nbf.v4.new_markdown_cell(md_eval),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_markdown_cell(md_final),
    nbf.v4.new_code_cell(code_final),
]

out_path = os.path.join(config.NOTEBOOK_DIR, "nlp_classification_notebook.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook → {out_path}")
