"""S4-S5. Dense / BM25 / hybrid (RRF) retrieval with optional cross-encoder rerank."""
from __future__ import annotations

from .embed import encode, load_embedder, load_index
from .sparse import BM25Index


def rrf(rankings, k: int = 60):
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: (-scores[c], c))  # ties broken by id for determinism


class Retriever:
    def __init__(self, cfg, chunks: dict[str, dict], passphrase=None):
        self.cfg, self.chunks = cfg, chunks
        self.index, self.ids = load_index(cfg, passphrase)
        self.embedder = load_embedder(cfg)
        self.bm25 = BM25Index(chunks)
        self._reranker = None

    def _passes(self, cid, filters):
        if not filters:
            return True
        c = self.chunks[cid]
        if filters.get("paper_ids") and c["paper_id"] not in filters["paper_ids"]:
            return False
        if filters.get("year_min") and (c["year"] or 0) < filters["year_min"]:
            return False
        if filters.get("year_max") and (c["year"] or 9999) > filters["year_max"]:
            return False
        return True

    def dense(self, query: str, k: int, filters=None):
        q = encode(self.embedder, [query], self.cfg, self.cfg.embedding.query_prefix)
        kk = min(len(self.ids), k * (5 if filters else 1))
        scores, idx = self.index.search(q, kk)
        res = [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
        res.sort(key=lambda x: (-x[1], x[0]))
        return [r for r in res if self._passes(r[0], filters)][:k]

    def sparse(self, query: str, k: int, filters=None):
        res = self.bm25.ranking(query, k * (5 if filters else 1))
        return [r for r in res if self._passes(r[0], filters)][:k]

    def rerank(self, query: str, cids: list[str]) -> list[str]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.cfg.retrieval.reranker)
        scores = self._reranker.predict([(query, self.chunks[c]["text"]) for c in cids])
        return [c for _, c in sorted(zip([-float(s) for s in scores], cids))]

    def retrieve(self, query: str, mode: str = "hybrid", rerank: bool | None = None,
                 final_k: int | None = None, filters=None) -> list[dict]:
        r = self.cfg.retrieval
        final_k = final_k or r.final_k
        rerank = r.rerank if rerank is None else rerank
        dense = [c for c, _ in self.dense(query, r.dense_k, filters)]
        if mode == "dense":
            fused = dense
        elif mode == "bm25":
            fused = [c for c, _ in self.sparse(query, r.sparse_k, filters)]
        else:
            sp = [c for c, _ in self.sparse(query, r.sparse_k, filters)]
            fused = rrf([dense, sp], r.rrf_k)
        pool = fused[: r.rerank_pool if rerank else final_k]
        if rerank and pool:
            pool = self.rerank(query, pool)
        return [dict(self.chunks[c], rank=i + 1) for i, c in enumerate(pool[:final_k])]
