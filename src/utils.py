"""Shared helpers: config loading, hashing, logging, text utilities, output cache."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class AttrDict(dict):
    """dict with attribute access (cfg.model, cfg.chunk.tokens)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(o):
    if isinstance(o, dict):
        return AttrDict({k: _wrap(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_wrap(v) for v in o]
    return o


def deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides: dict | None = None) -> AttrDict:
    import os

    path = Path(path or os.environ.get("RAG_CONFIG") or ROOT / "config" / "config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("extends", None)
    if base:  # e.g. extends: config.yaml (resolved next to this file)
        raw = deep_update(yaml.safe_load((path.parent / base).read_text(encoding="utf-8")) or {}, raw)
    if overrides:
        deep_update(raw, overrides)
    return _wrap(raw)


def rpath(cfg, key: str) -> Path:
    return ROOT / cfg.paths[key]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(cfg) -> str:
    return sha256_text(json.dumps(cfg, sort_keys=True, default=str))


def get_logger(name: str = "rag") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    try:
        (ROOT / "logs").mkdir(exist_ok=True)
        fh = logging.FileHandler(ROOT / "logs" / "rag.log", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:
        pass
    return log


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def norm_text(s: str) -> str:
    """Lowercase, unify quotes/dashes, collapse whitespace (used by validators)."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("\u00ad", "")
    return re.sub(r"\s+", " ", s).strip().lower()


class OutputCache:
    """Cache of model outputs keyed by a hash of prompt, model and options (B5)."""

    def __init__(self, directory, enabled: bool = True):
        self.dir = Path(directory)
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts) -> str:
        return sha256_text(json.dumps(parts, sort_keys=True, default=str))

    def get(self, key: str):
        if not self.enabled:
            return None
        p = self.dir / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["value"]
        return None

    def set(self, key: str, value) -> None:
        if self.enabled:
            (self.dir / f"{key}.json").write_text(json.dumps({"value": value}), encoding="utf-8")
