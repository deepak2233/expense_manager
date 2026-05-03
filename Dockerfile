# ─── Expense Classifier — Docker Image ───────────────────────────────────────────
# CPU-only PyTorch to keep image ~3GB instead of ~8GB
FROM python:3.12-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (CPU-only torch via extra-index-url)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Pre-download the HuggingFace model so the image works offline
RUN python -c "\
from transformers import pipeline; \
pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')"

# Copy project
COPY . .

# Create output dirs
RUN mkdir -p outputs/plots notebooks

# Env vars
ENV PROJECT_ROOT=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
