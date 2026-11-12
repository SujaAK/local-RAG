"""S4 (add-on A4). BM25 over chunk tokens."""
from __future__ import annotations

import re

from .utils import normalize_unicode

STOP = set("a an the of and or to in on for with by is are was were be been this that these those as at from it its "
           "we our which can not no but if than then also such into over".split())
_TOK = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(normalize_unicode(text).lower()) if t not in STOP]


class BM25Index:
    def __init__(self, chunks: dict[str, dict]):
        from rank_bm25 import BM25Okapi

        self.ids = sorted(chunks)
        self.bm25 = BM25Okapi([tokenize(chunks[i]["text"]) or ["_"] for i in self.ids])

    def ranking(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query) or ["_"])
        order = sorted(range(len(self.ids)), key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]