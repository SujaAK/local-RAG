"""S3. Local embeddings + exact FAISS inner-product index (normalised vectors)."""
from __future__ import annotations

import io

import numpy as np

from .store import close_db, dump_json, load_chunks, open_db, read_blob, write_blob
from .utils import get_logger, rpath

log = get_logger("embed")


def load_embedder(cfg):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg.embedding.model)


def encode(model, texts: list[str], cfg, prefix: str = "") -> np.ndarray:
    vecs = model.encode([prefix + t for t in texts], batch_size=cfg.embedding.batch_size,
                        normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_index(cfg, passphrase=None) -> int:
    import faiss

    conn = open_db(cfg, passphrase)
    chunks = load_chunks(conn)
    conn.close()
    ids = sorted(chunks)
    model = load_embedder(cfg)
    vecs = encode(model, [chunks[i]["text"] for i in ids], cfg, cfg.embedding.doc_prefix)
    sigma = float(cfg.privacy.embedding_noise_sigma)
    if sigma > 0:  # exploratory defence: noise on stored vectors only
        rng = np.random.default_rng(cfg.seed)
        vecs = vecs + rng.normal(0, sigma, vecs.shape).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    d = rpath(cfg, "index_dir")
    write_blob(d / "chunks.faiss", faiss.serialize_index(index).tobytes(), cfg, passphrase)
    write_blob(d / "ids.json", dump_json(ids), cfg, passphrase)
    log.info("indexed %d chunks (dim %d)", len(ids), vecs.shape[1])
    return len(ids)


def load_index(cfg, passphrase=None):
    import json

    import faiss

    d = rpath(cfg, "index_dir")
    raw = read_blob(d / "chunks.faiss", cfg, passphrase)
    index = faiss.deserialize_index(np.frombuffer(raw, dtype="uint8"))
    ids = json.loads(read_blob(d / "ids.json", cfg, passphrase))
    return index, ids
