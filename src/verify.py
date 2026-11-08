"""S9 (add-on A1). Local NLI verification: Supported / Unsupported / Contested / Insufficient."""
from __future__ import annotations

import re

import numpy as np

from .utils import split_sentences


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


FIGURE_RE = re.compile(r"^(figure|fig\.|table)\s*\d|\b(figure|fig\.|table)\s*\d+\s*:", re.I)


def usable_sentence(sent: str) -> bool:
    """False for PDF debris that must never be used as NLI evidence: heading fragments ('Citation recall.'),
    table rows and score lines ('Recall =66.6% Precision =50%'), figure captions and diagram text."""
    s = sent.strip()
    if len(s.split()) < 6:
        return False
    n = max(len(s), 1)
    if sum(ch.isalpha() for ch in s) / n < 0.6 or sum(ch.isdigit() for ch in s) / n > 0.15:
        return False
    if s.count("=") + s.count("%") >= 3 or sum(s.count(ch) for ch in "[]{}|") >= 6:
        return False
    if s[-1] not in ".?!:;)]\"\u201d\u2019'":              # cut off mid-sentence ('... its citation recall is 1 if')
        return False
    return not FIGURE_RE.search(s)


def top_sentences(claim: str, texts: list[str], n: int, embedder=None) -> list[str]:
    sents = [s for t in texts for s in split_sentences(t)]
    clean = [s for s in sents if usable_sentence(s)]
    sents = clean or sents                      # never return nothing just because everything looked noisy
    if not sents:
        return []
    if embedder is not None:
        vecs = embedder.encode([claim] + sents, normalize_embeddings=True, show_progress_bar=False)
        sims = vecs[1:] @ vecs[0]
    else:  # lexical fallback
        cw = set(claim.lower().split())
        sims = np.array([len(cw & set(x.lower().split())) / (len(cw) + 1) for x in sents])
    order = sorted(range(len(sents)), key=lambda i: (-float(sims[i]), i))[:n]
    return [sents[i] for i in sorted(order)]


def cap_words(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n])


def label_from_probs(p_ent, p_con, p_neu, p_con_other, tau_e, tau_c, has_evidence=True) -> str:
    if not has_evidence:
        return "Insufficient"
    if p_con_other >= tau_c:
        return "Contested"
    if p_ent >= tau_e:
        return "Supported"
    if p_con >= tau_c:
        return "Unsupported"
    return "Insufficient" if p_neu >= max(p_ent, p_con) else "Unsupported"


def build_premise(claim: str, texts: list[str], n: int, max_words: int, embedder=None) -> str:
    """The ONE premise builder: the n sentences of the cited texts most similar to the claim, capped.
    Used by S3 verification and by citation-metric scoring so both see the same evidence."""
    return cap_words(" ".join(top_sentences(claim, texts, n, embedder)), max_words)


class Verifier:
    def __init__(self, cfg, embedder=None, model: str | None = None, label_order: list[str] | None = None):
        from sentence_transformers import CrossEncoder

        self.cfg, self.embedder = cfg, embedder
        self.model_name = model or cfg.verify.model
        self.nli = CrossEncoder(self.model_name)
        self.order = [x.lower() for x in (label_order or cfg.verify.label_order)]

    def probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Returns array [n, 3] ordered as (contradiction, entailment, neutral)."""
        if not pairs:
            return np.zeros((0, 3))
        p = softmax(np.asarray(self.nli.predict(pairs)))
        return p[:, [self.order.index("contradiction"), self.order.index("entailment"),
                     self.order.index("neutral")]]

    def premise(self, claim: str, texts: list[str]) -> str:
        v = self.cfg.verify
        return build_premise(claim, texts, v.premise_sentences, v.premise_max_words, self.embedder)

    def score(self, claim: str, texts: list[str]):
        """(premise_used, probs[contra, entail, neutral]) under verify.premise_mode:
        'top_n' (default): the n most similar usable sentences, joined, as ONE premise;
        'max_sentence': each of the top `premise_pool` usable sentences is scored on its own and the best
        entailing one is used (the most contradicting one if nothing entails, so contradictions stay visible)."""
        v = self.cfg.verify
        if str(v.get("premise_mode", "top_n")) == "max_sentence":
            sents = top_sentences(claim, texts, int(v.get("premise_pool", 6)), self.embedder)
            if sents:
                ps = self.probs([(x, claim) for x in sents])
                i = int(ps[:, 1].argmax())
                if ps[i, 1] < 0.5:
                    i = int(ps[:, 0].argmax())
                return cap_words(sents[i], v.premise_max_words), ps[i]
        prem = self.premise(claim, texts)
        return prem, self.probs([(prem, claim)])[0]

    def entails(self, texts: list[str], hypothesis: str, tau: float | None = None) -> bool:
        """Entailment decision on the top-sentence premise. tau=None -> entailment must be the argmax
        (no dependence on S3's tuned thresholds; used by the independent check model)."""
        p = self.score(hypothesis, texts)[1]
        return bool(p[1] >= tau) if tau is not None else bool(p[1] == p.max())

    def _rank_candidates(self, claim: str, candidates: list[dict]) -> list[dict]:
        texts = [e["text"] for e in candidates]
        if self.embedder is not None:
            vecs = self.embedder.encode([claim] + texts, normalize_embeddings=True, show_progress_bar=False)
            sims = vecs[1:] @ vecs[0]
        else:
            cw = set(claim.lower().split())
            sims = np.array([len(cw & set(t.lower().split())) / (len(cw) + 1) for t in texts])
        order = sorted(range(len(candidates)), key=lambda i: (-float(sims[i]), i))
        return [candidates[i] for i in order]

    def _repair(self, claim: dict, candidates: list[dict], p_other: float, tau_e: float, tau_c: float,
                cited_papers: set):
        """Citation repair (system S3R): the claim failed against the passage it cited. Test the few most
        similar OTHER retrieved passages; if one entails it, re-point the citation to that passage.
        verify.repair_scope: 'same_paper' (default; only passages from the paper(s) the claim already cites,
        because NLI cannot tell WHICH paper a statement is about) or 'any'.
        Returns (result_or_None, trace) where trace lists every candidate tried with its NLI scores."""
        if str(self.cfg.verify.get("repair_scope", "same_paper")) == "same_paper" and cited_papers:
            candidates = [e for e in candidates if e.get("paper_id") in cited_papers]
        k = int(self.cfg.verify.get("repair_candidates", 3))
        best, trace = None, []
        for e in self._rank_candidates(claim["text"], candidates)[:k]:
            p = self.score(claim["text"], [e["text"]])[1]
            trace.append({"id": e["id"], "paper_id": e.get("paper_id"), "p_entail": round(float(p[1]), 3),
                          "p_contra": round(float(p[0]), 3)})
            if best is None or p[1] > best[1][1]:
                best = (e, p)
        if best is None:
            return None, trace
        e, p = best
        if p[1] >= tau_e and p[0] < tau_c:
            return {"p_contra": float(p[0]), "p_entail": float(p[1]), "p_neutral": float(p[2]),
                    "p_contra_other": p_other, "score": float(p[1]), "repaired_ids": [e["id"]],
                    "label": label_from_probs(p[1], p[0], p[2], p_other, tau_e, tau_c)}, trace
        return None, trace

    def verify_claims(self, claims: list[dict], evidence: list[dict], tau_e=None, tau_c=None,
                      repair: bool = False) -> list[dict]:
        v = self.cfg.verify
        tau_e = v.tau_entail if tau_e is None else tau_e
        tau_c = v.tau_contra if tau_c is None else tau_c
        by_id = {e["id"]: e for e in evidence}
        out = []
        for c in claims:
            ids = c.get("evidence_ids", [])
            cited = [by_id[i]["text"] for i in ids if i in by_id]
            others_ev = [e for e in evidence if e["id"] not in ids]
            others = [e["text"] for e in others_ev]
            p_other = 0.0
            if not cited:
                res = {"p_entail": 0.0, "p_contra": 0.0, "p_neutral": 1.0, "p_contra_other": 0.0,
                       "label": "Insufficient", "score": 0.0}
            else:
                p = self.score(c["text"], cited)[1]
                p_other = float(self.probs([(self.premise(c["text"], others), c["text"])])[0][0]) if others else 0.0
                res = {"p_contra": float(p[0]), "p_entail": float(p[1]), "p_neutral": float(p[2]),
                       "p_contra_other": p_other, "score": float(p[1]),
                       "label": label_from_probs(p[1], p[0], p[2], p_other, tau_e, tau_c)}
            if repair and res["label"] in ("Insufficient", "Unsupported") and others_ev:
                cited_papers = {by_id[i]["paper_id"] for i in ids if i in by_id}
                fixed, trace = self._repair(c, others_ev, p_other, tau_e, tau_c, cited_papers)
                res = fixed or res
                res["repair_trace"] = trace
            out.append(res)
        return out