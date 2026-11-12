"""Metric definitions (Part C4). Check exact ALCE definitions (R17) before writing them in the paper."""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

import numpy as np

from src.utils import norm_text


# ---------- retrieval ----------
def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm_text(s))


def gold_hit(chunk: dict, gold: dict, min_frac: float = 0.9) -> bool:
    """A chunk hits a gold span only if the span appears (near-)contiguously in the chunk:
    the longest contiguous run of matching word tokens must cover >= min_frac of the span.
    Shared common words never count (fixes the old bag-of-words 60% rule).
    Keep gold spans short (<= ~40 words) and inside one chunk or the chunk overlap."""
    if gold.get("paper_id") and chunk["paper_id"] != gold["paper_id"]:
        return False
    g, c = _tokens(gold["text"]), _tokens(chunk["text"])
    if not g:
        return False
    if " " + " ".join(g) + " " in " " + " ".join(c) + " ":
        return True
    m = SequenceMatcher(None, g, c, autojunk=False).find_longest_match(0, len(g), 0, len(c))
    return m.size >= max(1, math.ceil(min_frac * len(g)))


def retrieval_metrics(ranked: list[dict], gold: list[dict], ks=(5, 10)) -> dict:
    out = {}
    if not gold:
        return out
    hits = [[gold_hit(c, g) for g in gold] for c in ranked]
    for k in ks:
        top = hits[:k]
        out[f"recall@{k}"] = float(any(any(h) for h in top))
        out[f"allgold@{k}"] = float(all(any(h[j] for h in top) for j in range(len(gold))))
    rel = [any(h) for h in hits]
    out["mrr"] = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    # nDCG: each gold span is credited once (a chunk earns gain only if it covers a NOT-yet-covered span),
    # so DCG <= IDCG and nDCG <= 1 even when overlapping chunks hit the same span.
    covered, dcg = set(), 0.0
    for i, h in enumerate(hits[:10]):
        new = {j for j, x in enumerate(h) if x} - covered
        if new:
            dcg += 1.0 / math.log2(i + 2)
            covered |= new
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


# ---------- citation fidelity (ALCE-style, NLI based) ----------
def citation_scores(claims: list[dict], evidence_by_id: dict, entails) -> dict:
    """entails(list_of_cited_texts, claim) -> bool. The callable must build its premise the same way S3 does
    (top-sentence premise, see src.verify.build_premise), never a truncated concatenation.
    Recall: claim fully supported by its cited set.
    Precision (strict): a citation counts as relevant only if (a) the claim is supported by its full citation
    set and (b) the citation is not redundant (it supports the claim alone, or the other citations do not).
    Citations of unsupported claims and IDs that were never retrieved count as irrelevant. The literal ALCE
    rule only flags redundancy, so it scores every citation of an unsupported claim, and every lone citation,
    as relevant."""
    rec, prec_num, prec_den, fabricated = [], 0, 0, 0
    for c in claims:
        ids = list(dict.fromkeys(c.get("evidence_ids", [])))
        valid = [i for i in ids if i in evidence_by_id]
        fabricated += int(len(valid) != len(ids))
        prec_den += len(ids)
        if not valid:
            rec.append(0.0)
            continue
        texts = {i: evidence_by_id[i]["text"] for i in valid}
        full = bool(entails(list(texts.values()), c["text"]))
        rec.append(float(full))
        if not full:
            continue
        for i in valid:
            alone = entails([texts[i]], c["text"])
            rest = [texts[j] for j in valid if j != i]
            rest_ok = bool(rest) and entails(rest, c["text"])
            prec_num += int(alone or not rest_ok)
    n = len(claims)
    return {"citation_recall_auto": float(np.mean(rec)) if rec else float("nan"),
            "citation_precision_auto": prec_num / prec_den if prec_den else float("nan"),
            "fabrication_rate": fabricated / n if n else float("nan")}


def unsupported_rate(labels: list[str]) -> float:
    """Human labels of shown claims: Supported | Partly | Unsupported | Contradicted."""
    return sum(l in ("Unsupported", "Contradicted") for l in labels) / len(labels) if labels else float("nan")


def coverage(n_shown: int, n_generated: int) -> float:
    return n_shown / n_generated if n_generated else float("nan")


# ---------- verifier ----------
def risk_coverage(scores, is_bad, n_points: int = 101):
    """Sweep a threshold on verifier scores; risk = unsupported rate among kept claims."""
    s, b = np.asarray(scores, float), np.asarray(is_bad, float)
    if len(s) == 0:
        return [], [], float("nan")
    cov, risk = [], []
    for t in np.unique(np.concatenate([[s.min() - 1e-9], np.quantile(s, np.linspace(0, 1, n_points)), [s.max() + 1]])):
        keep = s >= t
        if keep.sum() == 0:
            continue
        cov.append(keep.mean())
        risk.append(b[keep].mean())
    order = np.argsort(cov)
    cov, risk = np.asarray(cov)[order], np.asarray(risk)[order]
    return cov.tolist(), risk.tolist(), float(np.sum((cov[1:] - cov[:-1]) * (risk[1:] + risk[:-1]) / 2)) if len(cov) > 1 else float("nan")


def ece(probs, correct, bins: int = 10) -> float:
    p, y = np.asarray(probs, float), np.asarray(correct, float)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def macro_f1(y_true, y_pred, labels) -> float:
    m, f1s = confusion(y_true, y_pred, labels), []
    for i in range(len(labels)):
        tp, fp, fn = m[i, i], m[:, i].sum() - m[i, i], m[i, :].sum() - m[i, i]
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


# ---------- abstention, conflict, determinism ----------
def abstention_prf(records: list[dict]) -> dict:
    """records: {answerable: bool, abstained: bool}. Positive class = should abstain (unanswerable)."""
    tp = sum(1 for r in records if not r["answerable"] and r["abstained"])
    fp = sum(1 for r in records if r["answerable"] and r["abstained"])
    fn = sum(1 for r in records if not r["answerable"] and not r["abstained"])
    ans = [r for r in records if r["answerable"]]
    return {"abstention_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "abstention_recall": tp / (tp + fn) if tp + fn else float("nan"),
            "false_abstention_rate": fp / len(ans) if ans else float("nan")}


def conflict_scores(result: dict, sides: list[str]) -> dict:
    """sides: paper_ids of the two disagreeing positions. Flagged = any Contested claim.
    Silent side-picking = answer is not flagged and cites only one side."""
    ev = {e["id"]: e["paper_id"] for e in result["evidence"]}
    shown = [result["claims"][i] for i in result.get("shown", [])]
    cited = {ev[i] for c in shown for i in c["evidence_ids"] if i in ev}
    flagged = any(c.get("label") == "Contested" for c in shown)
    both = all(s in cited for s in sides)
    return {"contested_flag": float(flagged),
            "silent_side_picking": float((not flagged) and (not both) and bool(cited)),
            "unflagged_contested": float(not flagged)}


def determinism_rate(outputs_by_run: list[list[str]]) -> float:
    """outputs_by_run[r][q]: share of questions whose outputs are byte-identical across all runs."""
    if not outputs_by_run:
        return float("nan")
    per_q = list(zip(*outputs_by_run))
    return sum(len(set(x)) == 1 for x in per_q) / len(per_q) if per_q else float("nan")