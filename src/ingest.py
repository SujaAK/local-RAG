"""S1. Ingest (synopsis stages 1-2): PDF -> pages -> headers stripped -> sections -> SQLite."""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from .store import close_db, open_db
from .utils import get_logger, rpath, sha256_file

log = get_logger("ingest")

HEADING_RE = re.compile(
    r"^(?:(\d+(?:\.\d+)*)\.?\s+)?(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"approach|experiments?|results?|evaluation|discussion|conclusions?|limitations|references|"
    r"bibliography|acknowledg(?:e)?ments|appendix)\b", re.I)
NUMBERED_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z\-,:& ]{2,80}$")
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def safe_id(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


HEADING_STOP = {"and", "of", "the", "for", "in", "on", "to", "a", "an", "with", "&", "vs", "via", "from", "by", "at"}


def _title_case_ok(words: list[str]) -> bool:
    """Words after a heading keyword must look like a title (Capitalised or a short stop word), not a sentence."""
    for w in words:
        if any(ch.isdigit() for ch in w):
            if len(w) > 3:                                # '85.2' is a table value; 'A', '2', 'A.1' are labels
                return False
            continue
        if not (w[:1].isupper() or not w[:1].isalpha() or w.lower() in HEADING_STOP):
            return False
    return True


def is_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    """A real section heading, not a body line that merely starts with a keyword such as 'results in ...'."""
    t = text.strip()
    if not t or len(t) > 100:
        return False
    if not (t[0].isupper() or t[0].isdigit()):          # headings never start in lower case
        return False
    words = t.split()
    if len(words) > 10 or t.endswith((",", ";")):
        return False
    if size >= body_size * 1.25 and len(t) < 80 and not t.endswith("."):
        return True
    if t.endswith(".") and len(words) > 3:               # a sentence, not a heading
        return False
    m = HEADING_RE.match(t)
    if m:
        ok = _title_case_ok(t[m.end():].split())
    elif NUMBERED_RE.match(t):
        ok = len(words) <= 9
    else:
        ok = False
    return ok and (bold or size >= body_size * 1.05 or t.isupper() or len(t) < 40)


def clean_heading(t: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", t.strip()).strip(" .:")[:80] or "Untitled"


def extract_pages(pdf_path: Path) -> list[dict]:
    import pymupdf as fitz  # PyMuPDF

    pages = []
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            height = page.rect.height
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for ln in block.get("lines", []):
                    spans = [s for s in ln["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join(s["text"].strip() for s in spans)
                    size = max(s["size"] for s in spans)
                    bold = any(s["flags"] & 16 for s in spans)
                    lines.append({"text": text, "size": size, "bold": bold,
                                  "y0": ln["bbox"][1], "y1": ln["bbox"][3]})
            pages.append({"page": pno, "height": height, "lines": lines})
    return pages


def strip_headers_footers(pages: list[dict], margin_frac=0.08, frac=0.4, min_pages=3) -> list[dict]:
    if len(pages) >= min_pages:
        counts = Counter()
        for p in pages:
            seen = set()
            for ln in p["lines"]:
                if ln["y1"] < p["height"] * margin_frac or ln["y0"] > p["height"] * (1 - margin_frac):
                    seen.add(re.sub(r"\d+", "#", ln["text"].strip().lower()))
            counts.update(seen)
        bad = {k for k, c in counts.items() if c >= frac * len(pages)}
    else:
        bad = set()
    for p in pages:
        keep = []
        for ln in p["lines"]:
            in_margin = ln["y1"] < p["height"] * margin_frac or ln["y0"] > p["height"] * (1 - margin_frac)
            key = re.sub(r"\d+", "#", ln["text"].strip().lower())
            if in_margin and (key in bad or re.fullmatch(r"#?\s*(page\s*)?#(\s*of\s*#)?", key)):
                continue
            keep.append(ln)
        p["lines"] = keep
    return pages


def build_sections(pages: list[dict]) -> list[dict]:
    sizes = [round(ln["size"], 1) for p in pages for ln in p["lines"] for _ in range(len(ln["text"]))]
    body = statistics.median(sizes) if sizes else 10.0
    sections, cur = [], {"name": "Front matter", "buf": "", "marks": []}
    for p in pages:
        new_page = True
        for ln in p["lines"]:
            t = ln["text"].strip()
            if is_heading(t, ln["size"], ln["bold"], body) and p["page"] >= 1:
                if cur["buf"].strip():
                    sections.append(cur)
                cur = {"name": clean_heading(t), "buf": "", "marks": []}
                new_page = True
                continue
            if new_page:
                cur["marks"].append([len(cur["buf"]), p["page"]])
                new_page = False
            if cur["buf"].endswith("-") and t[:1].islower():
                cur["buf"] = cur["buf"][:-1] + t
            else:
                cur["buf"] += (" " if cur["buf"] else "") + t
    if cur["buf"].strip():
        sections.append(cur)
    return sections


SKIP_AUTHOR_RE = re.compile(r"arxiv|abstract|preprint|submitted|copyright|©|doi:|https?:|"
                            r"^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|^\[?[a-z]{2}\.[A-Z]{2}\]?", re.I)


def guess_metadata(pages: list[dict], fallback_title: str = "") -> dict:
    meta = {"title": fallback_title, "authors": "", "year": None, "venue": ""}
    if not pages or not pages[0]["lines"]:
        return meta
    first = sorted((ln for ln in pages[0]["lines"] if ln["y0"] < pages[0]["height"] * 0.45),
                   key=lambda ln: (ln["y0"], ln["text"]))
    if first:
        mx = max(ln["size"] for ln in first)
        title_lines = [ln for ln in first if ln["size"] >= mx * 0.92]
        if title_lines:
            meta["title"] = " ".join(ln["text"] for ln in title_lines).strip()
            end_y = max(ln["y1"] for ln in title_lines)
            auth = []
            for ln in first:
                t = ln["text"].strip()
                if ln["y0"] < end_y or ln["size"] >= mx * 0.92:
                    continue
                if SKIP_AUTHOR_RE.search(t) or HEADING_RE.match(t) or NUMBERED_RE.match(t):
                    break  # stop at the arXiv stamp, abstract or first section heading
                auth.append(t)
                if len(auth) == 3:
                    break
            meta["authors"] = "; ".join(auth)[:300]
    text = " ".join(ln["text"] for ln in pages[0]["lines"])
    arx = re.search(r"arXiv:(\d{2})(\d{2})\.\d{4,5}", text)  # new-style arXiv ID encodes YYMM
    if arx and 7 <= int(arx.group(1)) <= 39:
        meta["year"] = 2000 + int(arx.group(1))
    else:
        ym = YEAR_RE.search(text)
        meta["year"] = int(ym.group(1)) if ym else None
    return meta


def read_overrides(papers_dir: Path) -> dict:
    p = papers_dir / "papers.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def ingest_all(cfg, passphrase=None, force: bool = False) -> dict:
    papers_dir = rpath(cfg, "papers_dir")
    pdfs = sorted(papers_dir.glob("*.pdf"))
    overrides = read_overrides(papers_dir)
    conn = open_db(cfg, passphrase)
    ok_meta = 0
    for pdf in pdfs:
        sha = sha256_file(pdf)
        pid = safe_id(pdf.stem)
        row = conn.execute("SELECT sha256 FROM papers WHERE paper_id=?", (pid,)).fetchone()
        if not force and row and row["sha256"] == sha:
            log.info("skip unchanged %s", pid)
            continue
        pages = strip_headers_footers(
            extract_pages(pdf), cfg.ingest.margin_frac, cfg.ingest.header_footer_frac,
            cfg.ingest.min_pages_for_hf)
        secs = build_sections(pages)
        meta = guess_metadata(pages, pdf.stem)
        ov = overrides.get(pdf.name)
        if ov:
            for k in ("title", "authors", "venue"):
                if ov.get(k):
                    meta[k] = ov[k]
            if ov.get("year"):
                meta["year"] = int(ov["year"])
        conn.execute("DELETE FROM sections WHERE paper_id=?", (pid,))
        conn.execute("DELETE FROM chunks WHERE paper_id=?", (pid,))
        conn.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?)",
                     (pid, sha, pdf.name, meta["title"], meta["authors"], meta["year"],
                      meta["venue"], len(pages)))
        for i, s in enumerate(secs):
            conn.execute("INSERT INTO sections VALUES (?,?,?,?,?)",
                         (pid, i, s["name"], s["buf"], json.dumps(s["marks"])))
        log.info("ingested %s: %d pages, %d sections", pid, len(pages), len(secs))
    rows = conn.execute("SELECT title, year FROM papers").fetchall()
    ok_meta = sum(1 for r in rows if r["title"] and r["year"])
    stats = {"papers": len(rows), "with_title_and_year": ok_meta,
             "pct_ok": round(100 * ok_meta / len(rows), 1) if rows else 0.0}
    log.info("metadata completeness: %s (DoD: >= 95%%)", stats)
    close_db(conn, cfg, passphrase)
    return stats