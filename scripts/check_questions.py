#!/usr/bin/env python
"""Check data/eval/questions.jsonl BEFORE any system is run on it (gate G1).

  python scripts/check_questions.py                  # validate every question
  python scripts/check_questions.py --dump 2305.14627   # print that paper's chunks (copy gold spans from here)

Why --dump: gold spans copied from a PDF viewer often differ from the extracted text (ligatures, hyphenation,
line breaks) and then never match. Copy the span from the dump instead; it is the text the system indexes.
Checks: required fields, unique ids, answerable/gold consistency, conflict_sides, multi-hop has >= 2 spans,
every gold span is found (>= 90% contiguous words) inside ONE chunk of its paper, span length <= 40 words.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import src` works from scripts/

TYPES = {"single_hop", "multi_hop", "conflict", "unanswerable", "draft"}
TARGETS = {"single_hop": 40, "multi_hop": 35, "conflict": 25, "unanswerable": 30, "draft": 20}  # whole set


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path.name} line {n}: invalid JSON ({e})")
            d["_line"] = n
            out.append(d)
    return out


def check(questions: list[dict], chunks: dict[str, dict], max_words: int = 40) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). `chunks` is {chunk_id: {paper_id, text, ...}}."""
    from eval.metrics import gold_hit

    by_paper: dict[str, list[dict]] = {}
    for c in chunks.values():
        by_paper.setdefault(c["paper_id"], []).append(c)
    errors, warns, seen_ids, seen_q = [], [], set(), set()
    for q in questions:
        tag = f"{q.get('id', '?')} (line {q.get('_line', '?')})"
        for k in ("id", "split", "type", "question", "answerable"):
            if k not in q:
                errors.append(f"{tag}: missing field '{k}'")
        if q.get("id") in seen_ids:
            errors.append(f"{tag}: duplicate id")
        seen_ids.add(q.get("id"))
        norm = " ".join(str(q.get("question", "")).lower().split())
        if norm in seen_q:
            warns.append(f"{tag}: duplicate question text")
        seen_q.add(norm)
        if q.get("split") not in ("dev", "test"):
            errors.append(f"{tag}: split must be 'dev' or 'test'")
        if q.get("type") not in TYPES:
            errors.append(f"{tag}: type must be one of {sorted(TYPES)}")
        if "EXAMPLE" in str(q.get("question", "")):
            errors.append(f"{tag}: still an EXAMPLE row; delete it")
        gold = q.get("gold_evidence", [])
        if q.get("answerable") is True and not gold:
            errors.append(f"{tag}: answerable but no gold_evidence")
        if q.get("answerable") is False and gold:
            errors.append(f"{tag}: unanswerable but has gold_evidence")
        if q.get("type") == "unanswerable" and q.get("answerable") is not False:
            errors.append(f"{tag}: type unanswerable needs answerable=false")
        if q.get("type") == "multi_hop" and len(gold) < 2:
            errors.append(f"{tag}: multi_hop needs >= 2 gold spans")
        if q.get("type") == "conflict":
            sides = q.get("conflict_sides", [])
            gp = {g.get("paper_id") for g in gold}
            if len(sides) != 2 or not set(sides) <= gp:
                errors.append(f"{tag}: conflict needs conflict_sides=[paperA, paperB], both with a gold span")
        for g in gold:
            pid, text = g.get("paper_id"), g.get("text", "")
            if pid not in by_paper:
                errors.append(f"{tag}: paper_id '{pid}' is not in the index")
                continue
            if len(text.split()) > max_words:
                warns.append(f"{tag}: gold span has {len(text.split())} words (> {max_words}); may cross a chunk boundary")
            if not any(gold_hit(c, g) for c in by_paper[pid]):
                errors.append(f"{tag}: gold span not found inside one chunk of {pid}: '{text[:60]}...' "
                              f"(copy it from --dump {pid})")
    return errors, warns


def summary(questions: list[dict]) -> str:
    cnt = Counter((q.get("split"), q.get("type")) for q in questions)
    tot = Counter(q.get("type") for q in questions)
    lines = ["type          dev  test  total  target"]
    for t in sorted(TYPES):
        lines.append(f"{t:<13} {cnt[('dev', t)]:>3}  {cnt[('test', t)]:>4}  {tot[t]:>5}  {TARGETS[t]:>6}")
    lines.append(f"{'ALL':<13} {sum(v for (s, _), v in cnt.items() if s == 'dev'):>3}  "
                 f"{sum(v for (s, _), v in cnt.items() if s == 'test'):>4}  {len(questions):>5}  {sum(TARGETS.values()):>6}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dump", default=None, metavar="PAPER_ID", help="print the chunks of one paper and exit")
    a = ap.parse_args()
    from src.store import close_db, load_chunks, open_db
    from src.utils import load_config, rpath

    cfg = load_config(a.config)
    conn = open_db(cfg)
    chunks = load_chunks(conn)
    close_db(conn, cfg)
    if a.dump:
        for cid, c in sorted(chunks.items()):
            if c["paper_id"] == a.dump:
                print(f"\n=== {cid} | {c['section']} | p.{c['page']} ===\n{c['text']}")
        return
    qs = load_jsonl(rpath(cfg, "eval_dir") / "questions.jsonl")
    errors, warns = check(qs, chunks)
    print(summary(qs))
    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\n{len(qs)} questions, {len(errors)} errors, {len(warns)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
