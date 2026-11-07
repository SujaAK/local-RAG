"""B8 (optional, only after gate G4). Finding hyperedges with verbatim-span check, stored in SQLite.
Stop rule: if extraction precision on the 5-paper pilot is below the pre-set threshold, do not use this."""
from __future__ import annotations

import json
import re

from .utils import norm_text, rpath

FIELDS = ["method", "task", "dataset", "metric", "value", "condition", "limitation"]
COMPARE = re.compile(r"\b(compare|compared|versus|vs\.?|better than|outperform\w*|difference between)\b", re.I)

SCHEMA = {"type": "object", "properties": {"findings": {"type": "array", "items": {
    "type": "object",
    "properties": {**{f: {"type": "string"} for f in FIELDS}, "span": {"type": "string"}},
    "required": FIELDS + ["span"]}}}, "required": ["findings"]}


def filter_by_span(tuples: list[dict], text: str) -> list[dict]:
    """Discard any tuple whose span is not found verbatim in the paper text."""
    nt = norm_text(text)
    return [t for t in tuples if t.get("span") and norm_text(t["span"]) in nt]


def extract_findings(cfg, model, section_text: str, cache) -> list[dict]:
    from .generate import _chat

    prompt = (rpath(cfg, "prompts_dir") / "extract_finding.txt").read_text(encoding="utf-8")
    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": section_text}]
    try:
        data = json.loads(_chat(cfg, model, msgs, SCHEMA, cache))
    except json.JSONDecodeError:
        return []
    return filter_by_span(data.get("findings", []), section_text)


def store_findings(conn, paper_id: str, chunk_id: str, findings: list[dict]) -> None:
    for f in findings:
        conn.execute(
            "INSERT INTO hyperedges(paper_id,chunk_id,method,task,dataset,metric,value,condition,limitation,span)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (paper_id, chunk_id, *[f.get(k, "") for k in FIELDS], f["span"]))


def route_to_graph(conn, query: str) -> list[str]:
    """Rule-based router: use graph only if the query names >= 2 known entities or uses comparison words."""
    ents = set()
    for row in conn.execute("SELECT method, task, dataset, metric FROM hyperedges"):
        ents.update(v.lower() for v in row if v and len(v) > 2)
    q = query.lower()
    hits = [e for e in ents if e in q]
    return hits if (len(hits) >= 2 or (hits and COMPARE.search(query))) else []


def expand_chunk_ids(conn, entities: list[str], limit: int = 8) -> list[str]:
    """1-hop expansion: chunks of hyperedges that mention any matched entity."""
    out: list[str] = []
    for e in entities:
        rows = conn.execute(
            "SELECT DISTINCT chunk_id FROM hyperedges WHERE lower(method)=? OR lower(task)=? "
            "OR lower(dataset)=? OR lower(metric)=? ORDER BY chunk_id", (e, e, e, e)).fetchall()
        out += [r["chunk_id"] for r in rows]
    return list(dict.fromkeys(out))[:limit]


def _aliases(cfg) -> dict[str, str]:
    """Optional alias table (config/aliases.yaml: canonical: [alias, ...]) for entity normalisation."""
    import yaml

    from .utils import ROOT

    p = ROOT / "config" / "aliases.yaml"
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {a.lower(): canon.lower() for canon, al in raw.items() for a in al}


def build_hypergraph(cfg, model: str, passphrase=None, max_papers: int | None = None) -> dict:
    """Extract Finding hyperedges per chunk (span must be verbatim in the chunk) and store them in SQLite."""
    from .store import close_db, load_chunks, open_db
    from .utils import OutputCache, get_logger

    log = get_logger("hypergraph")
    conn = open_db(cfg, passphrase)
    chunks = load_chunks(conn)
    cache = OutputCache(rpath(cfg, "runs_dir") / "cache",
                        enabled=cfg.privacy.cache_outputs and not cfg.privacy.encrypt_at_rest)
    papers = sorted({c["paper_id"] for c in chunks.values()})[:max_papers]
    alias = _aliases(cfg)
    n_edges = 0
    for pid in papers:
        conn.execute("DELETE FROM hyperedges WHERE paper_id=?", (pid,))
        for cid in (c for c in sorted(chunks) if chunks[c]["paper_id"] == pid):
            fs = extract_findings(cfg, model, chunks[cid]["text"], cache)
            for f in fs:
                for k in ("method", "task", "dataset", "metric"):
                    f[k] = alias.get(str(f.get(k, "")).lower(), str(f.get(k, "")).lower())
            store_findings(conn, pid, cid, fs)
            n_edges += len(fs)
        log.info("extracted hyperedges for %s", pid)
    close_db(conn, cfg, passphrase)
    return {"papers": len(papers), "hyperedges": n_edges}


def export_pilot_csv(cfg, path, passphrase=None, n_papers: int = 5) -> int:
    """Gate G4 pilot: dump extracted tuples for manual labelling (fill the `correct` column with 1/0)."""
    import csv

    from .store import open_db

    conn = open_db(cfg, passphrase)
    pids = [r[0] for r in conn.execute("SELECT DISTINCT paper_id FROM hyperedges ORDER BY paper_id LIMIT ?", (n_papers,))]
    rows = conn.execute(f"SELECT * FROM hyperedges WHERE paper_id IN ({','.join('?' * len(pids))}) ORDER BY edge_id",
                        pids).fetchall() if pids else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "paper_id", *FIELDS, "span", "correct"])
        for r in rows:
            w.writerow([r["edge_id"], r["paper_id"], *[r[k] for k in FIELDS], r["span"], ""])
    conn.close()
    return len(rows)


def pilot_precision(path) -> dict:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        labelled = [r for r in csv.DictReader(f) if r["correct"].strip() in ("0", "1")]
    n = len(labelled)
    return {"labelled": n, "precision": sum(r["correct"].strip() == "1" for r in labelled) / n if n else float("nan")}
