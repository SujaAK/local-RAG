#!/usr/bin/env python
"""Why does the NLI verifier reject a claim that a passage seems to support?  (offline, no LLM needed)

For one retrieved passage and one claim it prints the NLI scores for:
  * the claim as written, and plain rephrasings of it (to test whether the wording is the problem),
  * premises built from the 1, 2, 3 or 6 most similar sentences (to test the premise builder),
  * the whole passage, and every single sentence on its own (to find the sentence that really supports it).

Run from the project folder with the venv active and RAG_CONFIG set:
  python scripts/diag_nli.py --contains "3.3 Citation Quality"
  python scripts/diag_nli.py --contains "3.3 Citation Quality" --claim "your own claim here"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import src` works when run as scripts/diag_nli.py

DEFAULT_CLAIM = ("ALCE evaluates citation recall using the automatic citation recall metric, which determines "
                 "if the output is entirely supported by cited passages.")
DEFAULT_VARIANTS = [
    "Citation recall determines if the output is entirely supported by cited passages.",
    "ALCE measures citation recall by checking whether the output is fully supported by the passages it cites.",
]


def diagnose(verifier, claim: str, passage: str, variants=(), ns=(1, 2, 3, 6)) -> list[dict]:
    from src.utils import split_sentences
    from src.verify import build_premise

    rows = []
    sents = split_sentences(passage)
    for hyp in [claim, *variants]:
        for n in ns:
            prem = build_premise(hyp, [passage], n, 100_000, verifier.embedder)
            p = verifier.probs([(prem, hyp)])[0]
            rows.append({"hypothesis": hyp, "premise": f"top-{n} sentences", "p_entail": float(p[1]),
                         "p_contra": float(p[0]), "text": prem})
        p = verifier.probs([(passage, hyp)])[0]
        rows.append({"hypothesis": hyp, "premise": "whole passage", "p_entail": float(p[1]),
                     "p_contra": float(p[0]), "text": passage})
        if sents:
            ps = verifier.probs([(s, hyp) for s in sents])
            i = int(ps[:, 1].argmax())
            rows.append({"hypothesis": hyp, "premise": "best single sentence", "p_entail": float(ps[i][1]),
                         "p_contra": float(ps[i][0]), "text": sents[i]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contains", required=True, help="text that identifies the passage in the index")
    ap.add_argument("--claim", default=DEFAULT_CLAIM)
    ap.add_argument("--paper", default=None, help="restrict to one paper_id, e.g. 2305.14627")
    a = ap.parse_args(argv)

    import src  # noqa: F401  (sets offline mode for Hugging Face libraries)
    from src.embed import load_embedder
    from src.store import load_chunks, open_db
    from src.utils import load_config
    from src.verify import Verifier

    cfg = load_config()
    chunks = load_chunks(open_db(cfg))
    hits = [c for c in chunks.values() if a.contains.lower() in c["text"].lower()
            and (a.paper is None or c["paper_id"] == a.paper)]
    if not hits:
        print("No indexed passage contains that text.")
        return 1
    chunk = hits[0]
    print(f"Passage {chunk['chunk_id']} ({chunk['title']}, {chunk['section']}, p.{chunk['page']}); "
          f"{len(hits)} passage(s) matched, using the first.\n")
    v = Verifier(cfg, load_embedder(cfg))
    variants = DEFAULT_VARIANTS if a.claim == DEFAULT_CLAIM else []
    for r in diagnose(v, a.claim, chunk["text"], variants):
        print(f"entail={r['p_entail']:.3f} contra={r['p_contra']:.3f} | {r['premise']:<22} | claim: {r['hypothesis'][:70]}")
        print(f"      premise text: {' '.join(r['text'].split())[:300]}")
    print(f"\nThresholds in use: entail >= {cfg.verify.tau_entail}, contra >= {cfg.verify.tau_contra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())