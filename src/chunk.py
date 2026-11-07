"""S2. Section-aware chunking with overlap; chunk IDs are paper:section:index with char offsets."""
from __future__ import annotations

import bisect
import json
import re

from .store import close_db, open_db
from .utils import get_logger

log = get_logger("chunk")
_WORD = re.compile(r"\S+")


def chunk_text(text: str, tokens: int = 384, overlap: int = 48) -> list[tuple[int, int, str]]:
    """Window over whitespace tokens. Returns (char_start, char_end, text)."""
    if tokens <= overlap:
        raise ValueError("chunk tokens must exceed overlap")
    spans = [m.span() for m in _WORD.finditer(text)]
    if not spans:
        return []
    out, step, i = [], tokens - overlap, 0
    while i < len(spans):
        j = min(i + tokens, len(spans))
        s, e = spans[i][0], spans[j - 1][1]
        out.append((s, e, text[s:e]))
        if j == len(spans):
            break
        i += step
    return out


def page_for_offset(marks: list[list[int]], offset: int) -> int | None:
    if not marks:
        return None
    starts = [m[0] for m in marks]
    k = bisect.bisect_right(starts, offset) - 1
    return marks[max(k, 0)][1]


def chunk_all(cfg, passphrase=None) -> int:
    conn = open_db(cfg, passphrase)
    excl = {s.lower() for s in cfg.ingest.exclude_sections}
    conn.execute("DELETE FROM chunks")
    n = 0
    for sec in conn.execute("SELECT * FROM sections ORDER BY paper_id, idx").fetchall():
        if sec["name"].strip().lower() in excl:
            continue
        marks = json.loads(sec["page_marks"])
        for k, (s, e, txt) in enumerate(chunk_text(sec["text"], cfg.chunk.tokens, cfg.chunk.overlap)):
            cid = f"{sec['paper_id']}:{sec['idx']}:{k}"
            conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                         (cid, sec["paper_id"], sec["name"], page_for_offset(marks, s), s, e, txt))
            n += 1
    log.info("created %d chunks", n)
    close_db(conn, cfg, passphrase)
    return n
