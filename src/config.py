"""
Configuration — single source of truth for all magic strings, thresholds,
taxonomy definitions, and file paths.

Edit THIS file to change classification behaviour; never scatter constants
across multiple modules.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────────
# Support both local dev and Docker: PROJECT_ROOT is the expense_classifier/ dir
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebooks")

INPUT_FILE = os.path.join(RAW_DATA_DIR, "data.xlsx")
OUTPUT_CLASSIFIED = os.path.join(OUTPUT_DIR, "classified_data.xlsx")
OUTPUT_REVIEW_QUEUE = os.path.join(OUTPUT_DIR, "human_review_queue.xlsx")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "Executive_Summary.docx")

# ─── Taxonomy Specification ──────────────────────────────────────────────────────
# If two reviewers can't agree on a row given these definitions, they are too loose.
TAXONOMY = {
    "Equipment": (
        "A discrete, durable physical asset purchased as a single unit and "
        "typically inventoried (useful life > 1 year).  "
        "Examples: AC unit, dispenser, almirah, TV, projector, UPS, camera, "
        "chair, desk, kiosk, fire extinguisher, water cooler."
    ),
    "Material": (
        "A consumable, commodity, or bulk-purchased input that is installed/used "
        "as part of a larger system and is NOT individually tracked as an asset.  "
        "Examples: copper pipe, PVC pipe, cable, wire, insulation, valves, "
        "conduit, nut-bolts, clamps, patch cords."
    ),
    "Services": (
        "A labour-intensive engagement — professional work, installation, "
        "testing, commissioning, consultancy, or construction/civil/MEP work "
        "billed as a service.  "
        "Examples: consultancy fees, installation charges, civil work, "
        "carpentry, fire-fighting system commissioning, interior/MEP work."
    ),
    "OTHER": (
        "Accounting journal entries, provisions, reclassifications, transfers, "
        "or any remark that does NOT describe a procured good or service.  "
        "Must never be force-fit into the above three."
    ),
}

CANDIDATE_LABELS = ["Services", "Equipment", "Material"]

# ─── Journal-entry pre-filter (regex patterns) ──────────────────────────────────
JOURNAL_PATTERNS = [
    r"provision",
    r"reclass",
    r"transfer",
    r"trf\s+to",
    r"cwip",
    r"space\s+matrix\s+cost",
    r"exp\s+reclass",
]

# ─── Text normalisation: noise patterns to strip ────────────────────────────────
# Each tuple: (regex_pattern, replacement, use_re.IGNORECASE)
NOISE_PATTERNS = [
    (r"^\d{6,}\s*", "", True),
    (r"(WO|PO|WC|DOR|Dt|Inv\s*No)[\s:\-]*[\w/\-\.]+", "", True),
    (r"Period[\-\s]*\d{2}/\d{2}/\d{4}\s*to\s*\d{2}/\d{2}/\d{4}", "", True),
    (r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "", False),
    (r"\b\d{7,}\b", "", False),
    (r"received\s*\d+.*", "", True),
    (r"For\s*(T-\d+\s*)?For\s*Code.*", "", True),
    (r"\s+", " ", False),
]

# ─── Gold-standard evaluation rules (high-precision keywords) ────────────────────
GOLD_RULES = {
    "Equipment": [
        r"\bdispenser\b", r"\balmirah\b", r"\bcamera\b", r"\bdesktop\b",
        r"\blaptop\b", r"\bchair\b", r"\btable\b", r"\bmicrowave\b",
        r"\bfridge\b", r"\bups\b", r"\btv\b", r"\bprojector\b",
        r"\bkiosk\b", r"\bmouse\b", r"\brack\b", r"\bwater cooler\b",
        r"\bwheel chair\b", r"\bseesaw\b", r"\bslide\b",
        r"\bfire extinguisher\b", r"\bair cleaner\b", r"\bdashboard\b",
        r"\bpbx\b", r"\bscent diffuser\b", r"\bpallet\b",
    ],
    "Material": [
        r"\bpipe\b", r"\bcopper\b", r"\bwire\b", r"\bcable\b",
        r"\bpvc\b", r"\binsulation\b", r"\bswitch\b", r"\bsocket\b",
        r"\bpanel\b", r"\bvalve\b", r"\bconduit\b", r"\btray\b",
        r"\bnut\s*bolt", r"\bclamp\b", r"\banchor\b", r"\bmodule\b",
        r"\bpatch\s*cord", r"\bsfp\b",
    ],
    "Services": [
        r"\bconsultancy\b", r"\binstallation\s+charges\b",
        r"\blabour\b", r"\btesting\b.*\bcommissioning\b",
        r"\bcommissioning\b.*\btesting\b",
        r"\bcivil\s+work\b", r"\bcarpentry\b", r"\binterior\b.*\bwork\b",
        r"\bmep\s+work\b", r"\bfire\s+fighting\b",
        r"\bmaintenance\s+charges\b", r"\bupgradation\b",
        r"\bwiring\s+and\s+installation\b",
    ],
    "OTHER": [
        r"provision", r"reclass", r"trf\s+to", r"cwip",
    ],
}

# ─── Thresholds ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
LEARNABILITY_THRESHOLD = 0.85
MIN_TEXT_LENGTH = 3

# ─── Model ───────────────────────────────────────────────────────────────────────
ZERO_SHOT_MODEL = "typeform/distilbert-base-uncased-mnli"
