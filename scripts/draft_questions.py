#!/usr/bin/env python
"""Draft single-hop questions from your own index, then review them one keypress at a time.

  python scripts/draft_questions.py draft --n 30        # needs llama-server running (Terminal B)
  python scripts/draft_questions.py review              # y = keep, e = edit the question, n = skip, q = quit
  python scripts/draft_questions.py absent "dollar|usd|cost"   # is a term anywhere in the corpus? (unanswerable check)

How it works: a real sentence is picked from a chunk (one paper at a time, spread across papers); the local
model writes a question that the sentence answers; the SENTENCE is the gold span, so it is verbatim in your
index by construction. Nothing is added to questions.jsonl until you press y.

Disclose in the paper that single-hop questions were LLM-drafted from corpus sentences and then human-reviewed,
and that the drafting model was the same small model that is evaluated (a possible bias).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import src` works from scripts/

BAD_Q = re.compile(r"\b(this paper|the paper|the authors?|this work|this study|the study|we|our|us|these|those|"
                   r"the above|the following|here|they|sentence|figure|table|panel|cited|colored|dashed)\b", re.I)
SKIP_SECTIONS = {"front matter", "untitled", "references", "bibliography"}

SYSTEM = ("You write evaluation questions for a literature-review assistant. Given one sentence from a paper, "
          "write ONE standalone question that this sentence answers. Rules: name the specific method, dataset or "
          "topic explicitly; never say 'this paper', 'the authors', 'we' or 'the study'; do not copy the sentence; "
          "output only the question.")


def candidate_sentences(text: str, lo: int = 12, hi: int = 35) -> list[str]:
    from src.utils import split_sentences
    from src.verify import usable_sentence

    out = []
    for s in split_sentences(text):
        n = len(s.split())
        if lo <= n <= hi and s[0].isupper() and usable_sentence(s):
            out.append(s)
    return out


def names_something(q: str) -> bool:
    """A standalone question must name a method, dataset or system: an acronym, CamelCase name or number
    somewhere after the first word ('What is the target domain for the pretrained model?' names nothing)."""
    return any(any(ch.isupper() for ch in w) or any(ch.isdigit() for ch in w) for w in q.split()[1:])


def question_ok(q: str) -> bool:
    q = q.strip()
    return (len(q.split()) >= 6 and q.endswith("?") and not BAD_Q.search(q) and "\n" not in q
            and names_something(q))


def pick_sentences(chunks: dict[str, dict], n: int, seed: int) -> list[dict]:
    """Round-robin over papers so the draft is spread across the corpus."""
    rng = random.Random(seed)
    by_paper: dict[str, list[dict]] = {}
    for cid in sorted(chunks):
        c = chunks[cid]
        if c["section"].strip().lower() in SKIP_SECTIONS:
            continue
        for s in candidate_sentences(c["text"]):
            by_paper.setdefault(c["paper_id"], []).append({"paper_id": c["paper_id"], "title": c["title"],
                                                           "section": c["section"], "sentence": s})
    for v in by_paper.values():
        rng.shuffle(v)
    papers = sorted(by_paper)
    rng.shuffle(papers)
    picked, i = [], 0
    while len(picked) < n and any(by_paper.values()):
        p = papers[i % len(papers)]
        if by_paper[p]:
            picked.append(by_paper[p].pop())
        i += 1
    return picked


def next_id_number(path: Path) -> int:
    nums = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.search(r'"id"\s*:\s*"q(\d+)"', line)
            if m:
                nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def cmd_draft(a, cfg, chunks):
    from src.generate import _chat
    from src.utils import OutputCache, rpath

    out = rpath(cfg, "eval_dir") / "questions_draft.jsonl"
    cache = OutputCache(rpath(cfg, "runs_dir") / "cache", enabled=False)
    items, kept, tried = pick_sentences(chunks, a.n * 8, a.seed), [], 0
    for it in items:
        if len(kept) >= a.n:
            break
        tried += 1
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Paper title: {it['title']}\nSentence: {it['sentence']}\nQuestion:"}]
        try:
            q = _chat(cfg, cfg.model, msgs, None, cache).strip().splitlines()[0].strip().strip('"')
        except Exception as e:   # server not running, timeout, ...
            raise SystemExit(f"Model call failed ({e}). Is llama-server running in Terminal B?")
        if question_ok(q):
            kept.append({"split": a.split, "type": "single_hop", "answerable": True, "question": q,
                         "gold_evidence": [{"paper_id": it["paper_id"], "text": it["sentence"]}],
                         "source": "llm-drafted"})
            print(f"{len(kept):>3}/{a.n} {q}")
    with open(out, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(kept)} drafts to {out} ({tried} sentences tried). Next: python scripts/draft_questions.py review")


def cmd_review(a, cfg, chunks):
    from src.utils import rpath

    d = rpath(cfg, "eval_dir")
    draft, final = d / "questions_draft.jsonl", d / "questions.jsonl"
    if not draft.exists():
        raise SystemExit("No questions_draft.jsonl. Run the draft command first.")
    items = [json.loads(l) for l in draft.read_text(encoding="utf-8").splitlines() if l.strip()]
    num, kept, rest = next_id_number(final), 0, []
    for i, it in enumerate(items):
        g = it["gold_evidence"][0]
        print(f"\n[{i + 1}/{len(items)}] paper {g['paper_id']}\nQUESTION: {it['question']}\nEVIDENCE: {g['text']}")
        ans = input("y keep | e edit question | n skip | q quit > ").strip().lower()
        if ans == "q":
            rest = items[i:]
            break
        if ans == "e":
            new = input("new question: ").strip()
            if new:
                it["question"], ans = new, "y"
        if ans == "y":
            it["id"] = f"q{num:03d}"
            num += 1
            it["reviewed"] = True
            line = json.dumps({"id": it["id"], **{k: v for k, v in it.items() if k != "id"}}, ensure_ascii=False)
            text = final.read_text(encoding="utf-8") if final.exists() else ""
            with open(final, "a", encoding="utf-8") as f:
                f.write(("" if text.endswith("\n") or not text else "\n") + line + "\n")
            kept += 1
    draft.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rest), encoding="utf-8")
    print(f"\nKept {kept}. {len(rest)} drafts left. Now run: python scripts/check_questions.py")


def cmd_absent(a, cfg, chunks):
    rx = re.compile(a.pattern, re.I)
    hits = [(cid, c) for cid, c in sorted(chunks.items()) if rx.search(c["text"])]
    print(f"{len(hits)} of {len(chunks)} chunks match /{a.pattern}/")
    for cid, c in hits[:5]:
        m = rx.search(c["text"])
        s = max(0, m.start() - 80)
        print(f"  {cid}: ...{' '.join(c['text'][s:m.end() + 80].split())}...")
    print("-> safe as an unanswerable question only if none of these matches actually answers it." if hits
          else "-> no match anywhere: good candidate for an unanswerable question.")


def main():
    for stream in (sys.stdout, sys.stdin):   # Windows consoles: never crash on odd PDF characters
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--n", type=int, default=30)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--split", default="dev", choices=["dev", "test"])
    sub.add_parser("review")
    ab = sub.add_parser("absent")
    ab.add_argument("pattern")
    a = ap.parse_args()
    from src.store import close_db, load_chunks, open_db
    from src.utils import load_config

    cfg = load_config(a.config)
    conn = open_db(cfg)
    chunks = load_chunks(conn)
    close_db(conn, cfg)
    {"draft": cmd_draft, "review": cmd_review, "absent": cmd_absent}[a.cmd](a, cfg, chunks)


if __name__ == "__main__":
    main()