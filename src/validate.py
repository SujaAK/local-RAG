"""S8. Deterministic validators (plain code, no model).
V1 every cited ID was retrieved; V2 every claim has >= 1 ID;
V3 every number in a claim appears in the cited text; V4 every quoted string appears verbatim."""
from __future__ import annotations

import re

from .utils import norm_text

NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
QUOTE_RE = re.compile(r'"([^"]{4,})"|“([^”]{4,})”')


def _numbers(text: str) -> list[str]:
    out = []
    for m in NUM_RE.findall(text):
        n = m.replace(",", "").rstrip(".")
        out.append(n)
    return out


def _num_in(n: str, cited_norm: str) -> bool:
    flat = cited_norm.replace(",", "")
    return re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d]|\.\d)", flat) is not None


def validate_claim(claim: dict, evidence_by_id: dict[str, dict]) -> dict:
    ids = claim.get("evidence_ids", [])
    fails = []
    v2 = len(ids) >= 1
    if not v2:
        fails.append("V2: no evidence ID")
    bad = [i for i in ids if i not in evidence_by_id]
    v1 = not bad
    if not v1:
        fails.append(f"V1: cited ID(s) not retrieved: {bad}")
    cited = " ".join(norm_text(evidence_by_id[i]["text"]) for i in ids if i in evidence_by_id)
    missing_nums = [n for n in _numbers(claim["text"]) if not _num_in(n, cited)] if cited else []
    v3 = not missing_nums
    if not v3:
        fails.append(f"V3: number(s) not in cited text: {missing_nums}")
    quotes = [a or b for a, b in QUOTE_RE.findall(claim["text"])]
    missing_q = [q for q in quotes if norm_text(q) not in cited] if cited else []
    v4 = not missing_q
    if not v4:
        fails.append(f"V4: quote(s) not verbatim in cited text: {missing_q}")
    return {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "ok": not fails, "failures": fails}


def validate_all(claims: list[dict], evidence: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in evidence}
    return [validate_claim(c, by_id) for c in claims]
