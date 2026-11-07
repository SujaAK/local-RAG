"""S7. Generation: claim-form (structured output) and baseline (free text with [S#] citations)."""
from __future__ import annotations

import json
import re

from .utils import ROOT, OutputCache, rpath, split_sentences

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "evidence_ids"]}},
        "insufficient_evidence": {"type": "boolean"}},
    "required": ["claims", "insufficient_evidence"]}

_CITE = re.compile(r"\[(S\d+)\]")


def _prompt(cfg, name: str) -> str:
    return (rpath(cfg, "prompts_dir") / name).read_text(encoding="utf-8").strip()


def _messages(cfg, system_file: str, question: str, evidence_text: str):
    user = _prompt(cfg, "user_template.txt").format(evidence=evidence_text, question=question)
    return [{"role": "system", "content": _prompt(cfg, system_file)},
            {"role": "user", "content": user}]


def _chat(cfg, model, messages, fmt, cache: OutputCache) -> str:
    import ollama

    options = {"temperature": cfg.temperature, "seed": cfg.seed, "num_ctx": cfg.num_ctx,
               "num_predict": cfg.generation.num_predict}
    key = cache.key(model, messages, options, fmt)
    hit = cache.get(key)
    if hit is not None:
        return hit
    resp = ollama.chat(model=model, messages=messages, format=fmt, options=options)
    text = resp["message"]["content"]
    cache.set(key, text)
    return text


def generate_claims(cfg, model, question, evidence_text, cache) -> dict:
    raw = _chat(cfg, model, _messages(cfg, "system_claims.txt", question, evidence_text),
                CLAIM_SCHEMA, cache)
    try:
        data = json.loads(raw)
        claims = [{"text": str(c.get("text", "")).strip(),
                   "evidence_ids": [str(x) for x in c.get("evidence_ids", [])]}
                  for c in data.get("claims", []) if str(c.get("text", "")).strip()]
        return {"claims": claims, "insufficient_evidence": bool(data.get("insufficient_evidence")),
                "raw": raw}
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {"claims": [], "insufficient_evidence": True, "raw": raw, "parse_error": True}


def generate_baseline(cfg, model, question, evidence_text, cache) -> dict:
    """S0/S1: free text; sentences become claims so metrics are comparable across systems."""
    raw = _chat(cfg, model, _messages(cfg, "system_baseline.txt", question, evidence_text), None, cache)
    claims = []
    for sent in split_sentences(raw):
        ids = list(dict.fromkeys(_CITE.findall(sent)))
        text = _CITE.sub("", sent).strip()
        if text:
            claims.append({"text": text, "evidence_ids": ids})
    return {"claims": claims, "insufficient_evidence": False, "raw": raw}
