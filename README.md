# Expense Classification Pipeline

Classifies unstructured accounting expense remarks into **Services**, **Equipment**, **Material**, or **OTHER** (journal entries / out-of-scope).

---

## 🐳 Docker — One-Command Setup (Recommended for Handover)

```bash
# Build and run (first time ~5min for model download)
docker compose up --build

# Access the dashboard
open http://localhost:8501
```

That's it. The Docker image includes:
- All Python dependencies (CPU-only PyTorch — no GPU needed)
- Pre-cached HuggingFace model (works offline after build)
- Streamlit dashboard on port 8501

```bash
# Stop
docker compose down

# Rebuild after code changes
docker compose up --build

# Run in background
docker compose up -d
```

---

## 🖥️ Local Setup (without Docker)

```bash
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py

# OR run the CLI pipeline
python scripts/run_pipeline.py
```

---

## 🏗️ Pipeline Architecture

```
data.xlsx
    │
    ▼
┌─────────────────────┐
│  1. EDA             │  Constant cols, duplicates, journal entries
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. Preprocess      │  Normalise text → pre-filter journals → OTHER
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Classify        │  Zero-shot DistilBERT-MNLI
│                     │  Confidence < 0.55 → human review queue
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. Validate        │  Self-consistency · Learnability alarm · Gold eval
└────────┬────────────┘
         ▼
    outputs/
```

## ⚙️ Configuration

All tunable parameters live in `src/config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CONFIDENCE_THRESHOLD` | 0.55 | Below this → human review |
| `LEARNABILITY_THRESHOLD` | 0.85 | CV F1 alarm threshold |
| `ZERO_SHOT_MODEL` | distilbert-mnli | HuggingFace model |
| `JOURNAL_PATTERNS` | 7 regexes | Journal-entry detection |
| `NOISE_PATTERNS` | 8 regexes | Text normalisation rules |
