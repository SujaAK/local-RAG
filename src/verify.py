"""S9 (add-on A1). Local NLI verification: Supported / Unsupported / Contested / Insufficient."""
from __future__ import annotations

import numpy as np

from .utils import split_sentences


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def top_sentences(claim: str, texts: list[str], n: int, embedder=None) -> list[str]:
    sents = [s for t in texts for s in split_sentences(t)]
    if not sents:
        return []
    if embedder is not None:
        vecs = embedder.encode([claim] + sents, normalize_embeddings=True, show_progress_bar=False)
        sims = vecs[1:] @ vecs[0]
    else:  # lexical fallback
        cw = set(claim.lower().split())
        sims = np.array([len(cw & set(s.lower().split())) / (len(cw) + 1) for s in sents])
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

    def entails(self, texts: list[str], hypothesis: str, tau: float | None = None) -> bool:
        """Entailment decision on the top-sentence premise. tau=None -> entailment must be the argmax
        (no dependence on S3's tuned thresholds; used by the independent check model)."""
        p = self.probs([(self.premise(hypothesis, texts), hypothesis)])[0]
        return bool(p[1] >= tau) if tau is not None else bool(p[1] == p.max())

    def verify_claims(self, claims: list[dict], evidence: list[dict], tau_e=None, tau_c=None) -> list[dict]:
        v = self.cfg.verify
        tau_e = v.tau_entail if tau_e is None else tau_e
        tau_c = v.tau_contra if tau_c is None else tau_c
        by_id = {e["id"]: e for e in evidence}
        out = []
        for c in claims:
            cited = [by_id[i]["text"] for i in c.get("evidence_ids", []) if i in by_id]
            others = [e["text"] for e in evidence if e["id"] not in c.get("evidence_ids", [])]
            if not cited:
                out.append({"p_entail": 0.0, "p_contra": 0.0, "p_neutral": 1.0, "p_contra_other": 0.0,
                            "label": "Insufficient", "score": 0.0})
                continue
            p = self.probs([(self.premise(c["text"], cited), c["text"])])[0]
            p_other = float(self.probs([(self.premise(c["text"], others), c["text"])])[0][0]) if others else 0.0
            out.append({"p_contra": float(p[0]), "p_entail": float(p[1]), "p_neutral": float(p[2]),
                        "p_contra_other": p_other, "score": float(p[1]),
                        "label": label_from_probs(p[1], p[0], p[2], p_other, tau_e, tau_c)})
        return out
