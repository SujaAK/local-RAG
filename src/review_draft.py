"""S11. Draft-review mode: related-work list, overlap evidence, missing-topic suggestions, coverage.
No novelty score is produced (R24-R26)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .sparse import tokenize
from .utils import split_sentences


def review_draft(pipe, draft: str, filters=None) -> dict:
    cfg = pipe.cfg
    sents = split_sentences(draft)
    paper_scores: dict[str, float] = defaultdict(float)
    paper_meta, passages, overlaps = {}, [], []
    for s in sents:
        hits = pipe.retriever.retrieve(s, "hybrid", False, final_k=5, filters=filters)
        if not hits:
            continue
        vecs = pipe.retriever.embedder.encode([s] + [h["text"] for h in hits], normalize_embeddings=True,
                                              show_progress_bar=False)
        sims = vecs[1:] @ vecs[0]
        for h, sim in zip(hits, sims):
            paper_scores[h["paper_id"]] += 1.0 / (60 + h["rank"])
            paper_meta[h["paper_id"]] = (h["title"], h["year"])
            passages.append({"sentence": s, "chunk_id": h["chunk_id"], "paper_id": h["paper_id"],
                             "title": h["title"], "year": h["year"], "section": h["section"],
                             "page": h["page"], "similarity": round(float(sim), 3), "text": h["text"]})
            if sim >= cfg.review.overlap_sim:
                overlaps.append(passages[-1])
    ranked = sorted(paper_scores, key=lambda p: (-paper_scores[p], p))[: cfg.review.top_papers]
    related = [{"paper_id": p, "title": paper_meta[p][0], "year": paper_meta[p][1],
                "score": round(paper_scores[p], 4),
                "passages": [x for x in passages if x["paper_id"] == p][:3]} for p in ranked]
    # Missing topics: terms frequent in top related passages but absent from the draft (suggestions only).
    draft_terms = set(tokenize(draft))
    cnt = Counter(t for r in related for x in r["passages"] for t in tokenize(x["text"])
                  if t not in draft_terms and len(t) > 3 and not t.isdigit())
    missing = [t for t, _ in cnt.most_common(cfg.review.missing_terms)]
    n_papers = len({c["paper_id"] for c in pipe.chunks.values()})
    return {"related_work": related, "overlap_evidence": overlaps, "possible_missing_topics": missing,
            "coverage": f"Searched {n_papers} indexed papers; results cover only these papers.",
            "note": "Suggestions only. No novelty score is produced; this is not a plagiarism check."}
