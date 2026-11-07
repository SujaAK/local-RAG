#!/usr/bin/env bash
# SETUP PHASE (network allowed, run once). Check exact Ollama tags in the Ollama library.
# On CPU-only machines install CPU PyTorch FIRST, or sentence-transformers pulls a very large CUDA build:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
set -euo pipefail
ollama pull "${1:-qwen2.5:7b-instruct}"      # Tier 1: qwen2.5:3b-instruct
python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
CrossEncoder("cross-encoder/nli-deberta-v3-base")
CrossEncoder("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")   # independent NLI check model
print("models cached")
PY
