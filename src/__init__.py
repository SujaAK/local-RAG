"""local-rag package.

Offline by default: Hugging Face libraries are told not to contact the Hub (a cached model is enough) before
any of them is imported. Setting these variables here matters because newer transformers versions otherwise
make Hub requests even for cached models, which the offline guard (src/audit.py) correctly blocks.
To download models, run the download command directly (it does not import this package) or set
RAG_ALLOW_ONLINE=1 for one command.
"""
import os

if os.environ.get("RAG_ALLOW_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")