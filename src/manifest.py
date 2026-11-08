"""B5. Run manifest: everything needed to reproduce a run."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .utils import ROOT, config_hash, rpath, sha256_file, sha256_text


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown (not a git repo)"


def corpus_hash(cfg) -> str:
    hashes = sorted(sha256_file(p) for p in rpath(cfg, "papers_dir").glob("*.pdf"))
    return sha256_text("\n".join(hashes))


def model_digest(model: str, cfg=None) -> str:
    if cfg is not None and str(cfg.get("backend", "ollama")).lower() == "llamacpp":
        path = (cfg.get("llamacpp", {}) or {}).get("model_path")
        if path and Path(path).exists():
            return "sha256:" + sha256_file(path)
        return "unknown (set llamacpp.model_path to the .gguf file so the manifest can hash it)"
    try:
        import ollama

        for m in ollama.list()["models"]:
            name = m.get("model") or m.get("name")
            if name == model or name == f"{model}:latest":
                return m.get("digest", "unknown")
    except Exception:
        pass
    return "unknown"


def hardware() -> dict:
    info = {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
            "python": platform.python_version()}
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      text=True, stderr=subprocess.DEVNULL)
        info["gpu"] = out.strip()
    except Exception:
        info["gpu"] = "none detected"
    return info


def write_manifest(cfg, run_dir: Path, models: list[str], extra: dict | None = None) -> dict:
    lock = ROOT / "requirements.lock"
    m = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_hash": config_hash(cfg),
        "lock_file_hash": sha256_file(lock) if lock.exists() else "missing (run pip freeze > requirements.lock)",
        "llms": {mm: model_digest(mm, cfg) for mm in models},
        "embedding_model": cfg.embedding.model,
        "reranker_model": cfg.retrieval.reranker,
        "nli_model": cfg.verify.model,
        "seed": cfg.seed, "num_ctx": cfg.num_ctx, "temperature": cfg.temperature,
        "hardware": hardware(),
        "corpus_hash": corpus_hash(cfg),
        "config": json.loads(json.dumps(cfg, default=str)),
    }
    if extra:
        m.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m