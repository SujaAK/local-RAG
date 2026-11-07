"""S6. Evidence packing: IDs S1, S2, ... with title, year, section; fits the context budget."""
from __future__ import annotations

from .utils import approx_tokens


def pack_evidence(chunks: list[dict], budget_tokens: int):
    evidence, parts, used = [], [], 0
    for c in chunks:
        header_text = f"[S{len(evidence) + 1}] ({c['title']}, {c['year']}, section: {c['section']})\n{c['text']}"
        cost = approx_tokens(header_text)
        if used + cost > budget_tokens and evidence:
            break
        eid = f"S{len(evidence) + 1}"
        evidence.append({"id": eid, "chunk_id": c["chunk_id"], "paper_id": c["paper_id"],
                         "title": c["title"], "year": c["year"], "section": c["section"],
                         "page": c["page"], "text": c["text"]})
        parts.append(header_text)
        used += cost
    return evidence, "\n\n".join(parts)
