#!/usr/bin/env python
"""Download and verify open-access arXiv papers into data/papers and record metadata in papers.csv.

Run once while online (setup phase). Nothing here is used at query time.

Examples (from the project folder, venv active):
  python scripts/fetch_papers.py --email you@example.com                 # core papers + topic searches
  python scripts/fetch_papers.py --preset core --email you@example.com   # only the fixed reference list
  python scripts/fetch_papers.py --ids 2503.21322 2510.11394 --email you@example.com
  python scripts/fetch_papers.py --query "citation attribution" --total 30 --licence cc --email you@example.com
  python scripts/fetch_papers.py --dry-run --email you@example.com       # list what would be downloaded

What "verified" means here (each paper must pass all of these, otherwise it is rejected and logged):
  1. the arXiv API returns metadata for the ID (the paper exists, with title, authors, year);
  2. the download is a real PDF (starts with %PDF, not tiny);
  3. it has at least 2 pages and extractable text (not a scanned image);
  4. most words of the arXiv title appear on the PDF's first pages (the file matches its metadata);
  5. the licence is looked up from arXiv's OAI record and stored; with --licence cc only Creative Commons /
     public-domain papers are kept (needed if you plan to publish your corpus or benchmark).
Metadata written to papers.csv comes from the arXiv API, so titles/years are correct by construction.
Be polite to arXiv: this script waits ~3 s between requests and uses one connection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
OAI_BASES = ["https://oaipmh.arxiv.org/oai", "https://export.arxiv.org/oai2"]   # arXiv moved OAI-PMH in March 2025
LICENSE_HREF = re.compile(r'href="(https?://[^"]*(?:creativecommons\.org/(?:licenses|publicdomain)|arxiv\.org/licenses)[^"]*)"')
PDF = "https://export.arxiv.org/pdf/{id}"
DELAY = 3.2
CSV_FIELDS = ["filename", "title", "authors", "year", "venue", "arxiv_id", "licence"]

# Papers used or discussed in this project's plan (IDs checked during the literature scan, plus classics).
CORE_IDS = [
    "2005.11401",  # RAG (Lewis et al.)
    "2004.04906",  # Dense passage retrieval
    "2004.07180",  # SPECTER
    "2106.09685",  # LoRA
    "2403.05156",  # privacy of LLMs survey
    "2503.21322",  # HyperGraphRAG
    "2504.08758",  # Hyper-RAG
    "2510.11394",  # VeriCite
    "2510.02326",  # RA-FSM
    "2411.14199",  # OpenScholar
    "2409.13740",  # PaperQA2
    "2305.14627",  # ALCE
    "2402.15089",  # AttributionBench
    "2407.12861",  # CiteME
    "2210.13777",  # SciFact-Open
    "2502.11371",  # RAG vs GraphRAG
    "2506.05690",  # When to use graphs in RAG
    "2406.10251",  # quantization and RAG
    "2502.00306",  # membership inference (Riddle me this)
    "2405.20446",  # membership inference (IBM)
    "2502.11308",  # ALGEN embedding inversion
    "2602.03731",  # CUBO local RAG on laptops
    "2603.11513",  # small LMs and retrieved context
    "2605.20815",  # GraphRAG on consumer hardware
    "2606.25533",  # RAG security and privacy survey
    "2509.25498",  # NotebookLM-style overconfidence study
]
TOPIC_QUERIES = [
    "retrieval-augmented generation scientific literature",
    "citation attribution large language models verification",
    "scientific claim verification",
    "hallucination retrieval-augmented generation evaluation",
    "privacy retrieval-augmented generation",
    "local small language models retrieval-augmented generation",
]
DEFAULT_CATS = ["cs.CL", "cs.IR", "cs.AI", "cs.CR", "cs.LG"]

_last = [0.0]
UA = ["local-rag-corpus-fetcher/1.0"]


def _get(url: str, retries: int = 4, timeout: int = 90) -> bytes:
    for attempt in range(1, retries + 1):
        wait = DELAY - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA[0]})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 10 * attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def norm_id(s: str) -> str | None:
    s = s.strip()
    m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", s)
    if m:
        return m.group(1)
    m = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", s)
    return m.group(1) if m else None


def _ln(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_atom(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for entry in root:
        if _ln(entry.tag) != "entry":
            continue
        d = {"authors": [], "categories": []}
        for ch in entry:
            name = _ln(ch.tag)
            text = (ch.text or "").strip()
            if name == "id":
                d["abs_url"] = text
            elif name == "title":
                d["title"] = re.sub(r"\s+", " ", text)
            elif name == "published":
                d["published"] = text
            elif name == "author":
                for a in ch:
                    if _ln(a.tag) == "name":
                        d["authors"].append(re.sub(r"\s+", " ", (a.text or "").strip()))
            elif name == "journal_ref":
                d["journal_ref"] = re.sub(r"\s+", " ", text)
            elif name == "primary_category":
                d["primary"] = ch.attrib.get("term", "")
            elif name == "category":
                d["categories"].append(ch.attrib.get("term", ""))
        if "/api/errors" in d.get("abs_url", ""):
            continue
        aid = norm_id(d.get("abs_url", ""))
        if not aid or not d.get("title") or not d.get("published"):
            continue
        ver = re.search(r"v(\d+)$", d["abs_url"])
        d.update(id=aid, version=f"v{ver.group(1)}" if ver else "", year=int(d["published"][:4]))
        out.append(d)
    return out


def api_query(params: dict) -> list[dict]:
    return parse_atom(_get(API + "?" + urllib.parse.urlencode(params)))


def search(query: str, n: int, cats: list[str]) -> list[dict]:
    q = f'all:"{query}"'
    if cats:
        q += " AND (" + " OR ".join(f"cat:{c}" for c in cats) + ")"
    return api_query({"search_query": q, "start": 0, "max_results": n, "sortBy": "relevance"})


def by_ids(ids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 20):
        chunk = ids[i:i + 20]
        out += api_query({"id_list": ",".join(chunk), "max_results": len(chunk)})
    return out


_oai_fail = [0]


def get_licence(aid: str) -> str:
    """Licence URL from arXiv's OAI record; falls back to the abstract page. Fails fast (never stalls the run)."""
    q = urllib.parse.urlencode({"verb": "GetRecord", "identifier": f"oai:arXiv.org:{aid}", "metadataPrefix": "arXiv"})
    if _oai_fail[0] < 2:                       # after two failed papers, stop trying OAI for this run
        for base in OAI_BASES:
            try:
                root = ET.fromstring(_get(f"{base}?{q}", retries=1, timeout=25))
            except Exception:
                continue
            if any(_ln(el.tag) == "error" for el in root.iter()):
                continue
            _oai_fail[0] = 0
            for el in root.iter():
                if _ln(el.tag) == "license" and (el.text or "").strip():
                    return el.text.strip()
            return "arXiv-nonexclusive"        # no explicit licence element: default arXiv distribution licence
        _oai_fail[0] += 1
    try:                                       # fallback: licence link on the abstract page
        html = _get(f"https://arxiv.org/abs/{aid}", retries=1, timeout=25).decode("utf-8", "ignore")
    except Exception:
        return "unknown"
    m = LICENSE_HREF.search(html)
    return m.group(1) if m else "unknown"


def is_cc(licence: str) -> bool:
    return "creativecommons.org" in licence or "publicdomain" in licence


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


def verify_pdf(path: Path, title: str) -> tuple[bool, str]:
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        return False, "not a PDF (missing %PDF header)"
    if len(data) < 10_000:
        return False, f"file too small ({len(data)} bytes)"
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as e:      # corrupt file
        return False, f"cannot open: {e}"
    with doc:
        if doc.page_count < 2:
            return False, "fewer than 2 pages"
        text = " ".join(doc[i].get_text() for i in range(min(3, doc.page_count)))
    if len(text) < 1500:
        return False, "little extractable text (scanned or empty)"
    want = _tokens(title)
    if want and len(want & _tokens(text)) / len(want) < 0.6:
        return False, "title words not found on first pages (file may not match metadata)"
    return True, "ok"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f) if r.get("filename")}


def write_csv(path: Path, rows: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in CSV_FIELDS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["core", "topics", "all"], default=None)
    ap.add_argument("--ids", nargs="*", default=[], help="explicit arXiv IDs or URLs")
    ap.add_argument("--query", action="append", default=[], help="search phrase (repeatable)")
    ap.add_argument("--per-query", type=int, default=15, help="candidates fetched per query")
    ap.add_argument("--total", type=int, default=60, help="stop after this many papers are in papers.csv")
    ap.add_argument("--since-year", type=int, default=2020)
    ap.add_argument("--cats", default=",".join(DEFAULT_CATS), help="comma-separated arXiv categories ('' for any)")
    ap.add_argument("--licence", choices=["any", "cc"], default="any",
                    help="'cc' keeps only Creative Commons / public domain papers (needed to publish the corpus)")
    ap.add_argument("--dest", default="data/papers")
    ap.add_argument("--email", default="", help="contact address added to the User-Agent (polite to arXiv)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.email:
        UA[0] = f"local-rag-corpus-fetcher/1.0 (mailto:{a.email})"
    preset = a.preset or ("all" if not a.ids and not a.query else None)
    ids = [norm_id(x) for x in a.ids]
    if None in ids:
        print("Could not parse one of the --ids values.", file=sys.stderr)
        return 2
    queries = list(a.query)
    if preset in ("core", "all"):
        ids += CORE_IDS
    if preset in ("topics", "all"):
        queries += TOPIC_QUERIES
    cats = [c.strip() for c in a.cats.split(",") if c.strip()]

    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "papers.csv"
    rows = read_csv(csv_path)
    have = {r.get("arxiv_id") for r in rows.values() if r.get("arxiv_id")}
    report = []

    candidates: dict[str, dict] = {}
    print(f"Fetching metadata for {len(ids)} fixed IDs and {len(queries)} queries ...")
    for meta in by_ids(list(dict.fromkeys(ids))) if ids else []:
        candidates.setdefault(meta["id"], meta)
    for aid in set(ids) - set(candidates):
        report.append({"arxiv_id": aid, "status": "rejected", "reason": "no metadata returned by arXiv API"})
    for q in queries:
        for meta in search(q, a.per_query, cats):
            if meta["year"] >= a.since_year:
                candidates.setdefault(meta["id"], meta)
    print(f"{len(candidates)} candidate papers.")

    added = 0
    for n_done, (aid, meta) in enumerate(candidates.items(), start=1):
        print(f"[{n_done}/{len(candidates)}] {aid}", flush=True)
        if len(rows) >= a.total:
            print(f"Reached --total {a.total}; stopping.")
            break
        fname = aid.replace("/", "_") + ".pdf"
        if aid in have or fname in rows:
            report.append({"arxiv_id": aid, "status": "skipped", "reason": "already in papers.csv", "title": meta["title"]})
            continue
        if a.dry_run and a.licence == "any":
            print(f"  would download {aid}  {meta['year']}  {meta['title'][:70]}  [licence not checked in dry run]")
            continue
        licence = get_licence(aid)
        if a.licence == "cc" and not is_cc(licence):
            report.append({"arxiv_id": aid, "status": "rejected", "reason": f"licence not CC: {licence}",
                           "licence": licence, "title": meta["title"]})
            continue
        if a.dry_run:
            print(f"  would download {aid}  {meta['year']}  {meta['title'][:70]}  [{licence}]")
            continue
        target = dest / fname
        tmp = dest / (fname + ".part")
        try:
            tmp.write_bytes(_get(PDF.format(id=aid)))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            report.append({"arxiv_id": aid, "status": "rejected", "reason": f"download failed: {e}", "title": meta["title"]})
            print(f"  FAIL {aid}: download failed ({e})")
            continue
        ok, why = verify_pdf(tmp, meta["title"])
        if not ok:
            tmp.unlink(missing_ok=True)
            report.append({"arxiv_id": aid, "status": "rejected", "reason": why, "licence": licence, "title": meta["title"]})
            print(f"  REJECT {aid}: {why}")
            continue
        tmp.replace(target)
        venue = meta.get("journal_ref") or f"arXiv:{aid} [{meta.get('primary', '')}]"
        rows[fname] = {"filename": fname, "title": meta["title"], "authors": "; ".join(meta["authors"]),
                       "year": meta["year"], "venue": venue, "arxiv_id": aid, "licence": licence}
        write_csv(csv_path, rows)      # saved after every paper, so an interruption loses nothing
        report.append({"arxiv_id": aid, "status": "accepted", "reason": "ok", "licence": licence,
                       "title": meta["title"], "version": meta["version"], "sha256": sha256_file(target)})
        added += 1
        print(f"  OK {aid}  {meta['year']}  {meta['title'][:70]}")

    if not a.dry_run:
        rp = dest / "fetch_report.csv"
        keys = ["arxiv_id", "status", "reason", "licence", "version", "sha256", "title"]
        with open(rp, "a" if rp.exists() else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if f.tell() == 0:
                w.writeheader()
            w.writerows(report)
    summary = {s: sum(1 for r in report if r["status"] == s) for s in ("accepted", "rejected", "skipped")}
    print(json.dumps({**summary, "papers_in_csv": len(rows), "dry_run": a.dry_run}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())