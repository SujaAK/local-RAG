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


def _chat_ollama(cfg, model, messages, fmt, options) -> str:
    import ollama

    resp = ollama.chat(model=model, messages=messages, format=fmt, options=options)
    return resp["message"]["content"]


def _chat_llamacpp(cfg, model, messages, fmt) -> str:
    """llama.cpp `llama-server` (OpenAI-compatible). Context size is set when the server starts (-c).
    Uses only the standard library; the proxy handler is emptied so Windows system proxies never see
    the localhost request."""
    import urllib.request

    lc = cfg.get("llamacpp", {}) or {}
    url = str(lc.get("url", "http://127.0.0.1:8080")).rstrip("/") + "/v1/chat/completions"
    body = {"model": model, "messages": messages, "temperature": cfg.temperature, "seed": cfg.seed,
            "max_tokens": cfg.generation.num_predict, "stream": False}
    if fmt:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "output", "strict": True, "schema": fmt}}
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=float(lc.get("timeout", 900))) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


_IDENT: dict[str, str] = {}


def server_identity(cfg) -> str:
    """What llama-server has actually loaded (model file + context size), asked from the server itself.
    Used in the output-cache key and the run manifest, because `model` in the config is only a label and
    swapping the .gguf (e.g. 1.5B -> 3B) must never return the old model's cached answers."""
    import urllib.request

    base = str((cfg.get("llamacpp", {}) or {}).get("url", "http://127.0.0.1:8080")).rstrip("/")
    if base in _IDENT:
        return _IDENT[base]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    ident = "unknown"
    for path in ("/props", "/v1/models"):
        try:
            with opener.open(base + path, timeout=5) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
        if path == "/props":
            mp = d.get("model_path") or d.get("model_alias")
            n_ctx = (d.get("default_generation_settings") or {}).get("n_ctx")
            if mp:
                ident = f"{mp}|n_ctx={n_ctx}"
        else:
            m = (d.get("data") or [{}])[0]
            meta = m.get("meta") or {}
            if m.get("id"):
                ident = f"{m['id']}|params={meta.get('n_params')}|size={meta.get('size')}"
        if ident != "unknown":
            break
    if ident != "unknown":
        _IDENT[base] = ident
    return ident


def _chat(cfg, model, messages, fmt, cache: OutputCache) -> str:
    backend = str(cfg.get("backend", "ollama")).lower()
    options = {"temperature": cfg.temperature, "seed": cfg.seed, "num_ctx": cfg.num_ctx,
               "num_predict": cfg.generation.num_predict}
    ident = server_identity(cfg) if backend == "llamacpp" else ""
    key = cache.key(backend, model, ident, messages, options, fmt)
    hit = cache.get(key)
    if hit is not None:
        return hit
    if backend == "llamacpp":
        text = _chat_llamacpp(cfg, model, messages, fmt)
    elif backend == "ollama":
        text = _chat_ollama(cfg, model, messages, fmt, options)
    else:
        raise ValueError(f"unknown backend: {backend!r} (use 'ollama' or 'llamacpp')")
    cache.set(key, text)
    return text


def generate_claims(cfg, model, question, evidence_text, cache) -> dict:
    # the claims prompt is selectable (config key `claims_prompt`) so prompt variants can be compared as ablations
    raw = _chat(cfg, model, _messages(cfg, str(cfg.get("claims_prompt", "system_claims.txt")), question, evidence_text),
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