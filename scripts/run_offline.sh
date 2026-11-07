#!/usr/bin/env bash
# RUN PHASE helper. Usage:
#   bash scripts/run_offline.sh <command...>            # offline env vars + localhost-only model server
#   sudo -E bash scripts/run_offline.sh --netns "<command>"   # T11: no network interfaces at all (Linux)
# In --netns mode an Ollama server is started INSIDE the namespace (the host's one is unreachable there);
# models are read from $OLLAMA_MODELS (default ~/.ollama/models; with sudo pass -E and set OLLAMA_MODELS).
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OLLAMA_HOST=127.0.0.1:11434
if [[ "${1:-}" == "--netns" ]]; then
  shift
  export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
  exec unshare -n bash -c "ip link set lo up; ollama serve >/tmp/ollama-ns.log 2>&1 & sleep 5; $1"
fi
exec "$@"
